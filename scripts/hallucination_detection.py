import os
import json
import argparse
import torch
from tqdm import tqdm
import time
import gc

from src.data_loader import load_structured_qa_dataset
from src.answer_generator import AnswerGenerator
from src.detection_methods.hallucination_detection.selfcheckgpt_wrapper import SelfCheckGPTWrapper
from src.detection_methods.hallucination_detection.semantic_entropy import SemanticEntropyCalculator
from src.detection_methods.hallucination_detection.seu_calculator import SemanticEmbeddingUncertaintyCalculator
from src.detection_methods.hallucination_detection.sindex_calculator import SIndexCalculator
from src.detection_methods.hallucination_detection.mind_wrapper import MINDWrapper
from src.detection_methods.hallucination_detection.saplma_wrapper import SAPLMAWrapper
from src.detection_methods.hallucination_detection.eubhd_calculator import EUBHDCalculator
from src.llm_judge import LLMJudge
from src.hallucination_pipeline import run_pipeline
from src.utils import save_results_jsonl

def sanitize_path_component(name: str) -> str:
    if name is None:
        return "none"
    return name.replace('/', '_').replace('-', '_').replace('.', '_').lower()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Hallucination Benchmark Experiment")
    # Core Model & Generation Args
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model for answer generation.")
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed for generation.")
    parser.add_argument("--device", type=str, default="auto", choices=['cuda', 'cpu', 'auto'], help="Device for answer generator, NLI model & Embedding models.")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Max new tokens for ALL generations.")
    parser.add_argument("--main_temp", type=float, default=0.8, help="Temperature for MAIN answer generation.")
    parser.add_argument("--main_top_p", type=float, default=0.9, help="Top-p for MAIN answer generation.")
    parser.add_argument("--sample_temp", type=float, default=0.8, help="Temperature for ADDITIONAL sample generation.")
    parser.add_argument("--sample_top_p", type=float, default=0.9, help="Top-p for ADDITIONAL sample generation.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of additional samples for methods like SelfCheckGPT, PTrue, SemanticEntropy, SEU, SIndex.")

    # Judge Args
    parser.add_argument("--judge_model_name_or_path", type=str, required=True, help="Model for LLM-as-a-judge.")
    parser.add_argument("--judge_device", type=str, default="auto", choices=['cuda', 'cpu', 'auto'], help="Device for judge model.")
    parser.add_argument("--cache_base_dir", type=str, default="cache", help="Base directory to store judge caches.")
    parser.add_argument("--judge_max_retries", type=int, default=5, help="Max retries for LLM judge if parsing fails.")
    parser.add_argument("--judge_retry_delay", type=int, default=0, help="Delay (seconds) between LLM judge retries.")

    # Data and Output Args
    parser.add_argument("--data_base_dir", type=str, default="data", help="Base directory where dataset-specific subdirectories are located.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=['hotpotqa', 'popqa', 'complexwebq', '2wikimultihopqa','nq','triviaqa'], help="Name of the dataset to load.")
    parser.add_argument("--dataset_type", type=str, default="total", help="Type of QA to filter (e.g., 'bridge', 'comparison', 'total').")
    parser.add_argument("--results_base_dir", type=str, default="results", help="Base directory to save results and metrics.")
    parser.add_argument("--start_question", type=int, default=0, help="Start question to process (for testing).")
    parser.add_argument("--end_question", type=int, default=500, help="End question to process (for testing).")

    # Detection Method Args
    parser.add_argument(
        "--detect_methods", nargs='+',
        default=['mqag', 'bertscore', 'ngram', 'nli', 'ptrue', 'lnpp', 'lnpe', 'semantic_entropy', 'seu', 'sindex'],
        choices=[
            'mqag', 'bertscore', 'ngram', 'nli', 'llm_prompt', 'ptrue',
            'lnpp', 'lnpe', 'semantic_entropy', 'seu', 'sindex', 'mind', 'saplma', 'eubhd'
        ],
        help="List of detection methods to run."
    )
    # Method Specific Args
    parser.add_argument("--sindex_embedding_model", type=str, default=None, help="Embedding model for SIndex. Defaults to 'all-MiniLM-L6-v2'.")
    parser.add_argument("--sindex_threshold", type=float, default=0.95, help="Cosine similarity threshold for SIndex clustering (0 to 1).")

    parser.add_argument("--split_sentences_for_detection", action='store_true', help="Split main answer into sentences for *SelfCheckGPT* detection methods.")
    parser.add_argument("--aggregation_strategy", type=str, default="max", choices=['max', 'avg'], help="How to aggregate *sentence* scores for *SelfCheckGPT* methods.")

    parser.add_argument("--mind_classifier_path", type=str, default=None, help="Path to the pre-trained MIND MLP classifier checkpoint (.pt file).")

    parser.add_argument("--saplma_probe_path", type=str, default=None, help="Path to the pre-trained SAPLMA probe checkpoint (.pt file).")
    parser.add_argument("--saplma_layer", type=int, default=-1, help="Layer index for SAPLMA probe.")

    parser.add_argument("--eubhd_idf_path", type=str, default=None, help="Path to the pickled token IDF numpy array (.pkl file) for EUBHD.")
    parser.add_argument("--eubhd_gamma", type=float, default=0.9, help="Discount factor (gamma) for EUBHD.")
    parser.add_argument("--eubhd_no_penalty", action='store_true', help="Disable EUBHD penalty transmission.")
    parser.add_argument("--eubhd_all_tokens", action='store_true', help="Apply EUBHD scoring to all tokens.")
    parser.add_argument("--eubhd_no_idf", action='store_true', help="Disable IDF weighting in EUBHD.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    script_start_time = time.time()
    print("--- Starting Hallucination Detection Experiment ---")

    sane_target_llm_name = sanitize_path_component(args.model_name_or_path)
    sane_judge_llm_name = sanitize_path_component(args.judge_model_name_or_path)
    sane_dataset_name = sanitize_path_component(args.dataset_name)
    sane_dataset_type = sanitize_path_component(args.dataset_type)

    print(f"Target LLM: {args.model_name_or_path} (Sanitized: {sane_target_llm_name})")
    print(f"Judge LLM: {args.judge_model_name_or_path} (Sanitized: {sane_judge_llm_name})")
    print(f"Dataset: {args.dataset_name} (Sanitized: {sane_dataset_name})")
    print(f"Dataset Type: {args.dataset_type} (Sanitized: {sane_dataset_type})")
    print(f"Seed: {args.base_seed}")

    print("Full Args (excluding base paths shown above):")
    args_to_print = vars(args).copy()
    for key_to_pop in ['data_base_dir', 'cache_base_dir', 'results_base_dir',
                       'model_name_or_path', 'judge_model_name_or_path',
                       'dataset_name', 'dataset_type', 'base_seed']:
        args_to_print.pop(key_to_pop, None)
    print(json.dumps(args_to_print, indent=2))

    current_run_judge_cache_dir = os.path.join(
        args.cache_base_dir, sane_target_llm_name, sane_dataset_name,
        sane_dataset_type, sane_judge_llm_name
    )
    current_run_judge_cache_file = os.path.join(current_run_judge_cache_dir, "judge_cache.json")

    current_run_output_dir = os.path.join(
        args.results_base_dir, sane_target_llm_name, sane_dataset_name, sane_dataset_type
    )

    print(f"Judge cache for this run: {current_run_judge_cache_file}")
    print(f"Results directory for this run: {current_run_output_dir}")

    try:
        os.makedirs(args.cache_base_dir, exist_ok=True)
        os.makedirs(args.results_base_dir, exist_ok=True)
        os.makedirs(current_run_judge_cache_dir, exist_ok=True)
        os.makedirs(current_run_output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error setting up output/cache directories: {e}")
        return

    if 'mind' in args.detect_methods and (not args.mind_classifier_path or not os.path.exists(args.mind_classifier_path)):
        print(f"ERROR: --mind_classifier_path '{args.mind_classifier_path}' required and must exist for 'mind' method.")
        return
    if 'saplma' in args.detect_methods and (not args.saplma_probe_path or not os.path.exists(args.saplma_probe_path)):
        print(f"ERROR: --saplma_probe_path '{args.saplma_probe_path}' required and must exist for 'saplma' method.")
        return
    if 'eubhd' in args.detect_methods and (not args.eubhd_no_idf and (not args.eubhd_idf_path or not os.path.exists(args.eubhd_idf_path))):
        print(f"ERROR: EUBHD IDF weighting enabled, but --eubhd_idf_path '{args.eubhd_idf_path}' is required and must exist.")
        return

    print("\nLoading Data...")
    dataset = []
    llm_judge = None
    answer_generator = None
    selfcheck_wrapper = None
    se_calculator = None
    seu_calculator = None
    sindex_calculator = None
    mind_wrapper = None
    saplma_wrapper = None
    eubhd_calculator = None

    try:
        loaded_data_map = load_structured_qa_dataset(
            dataset_name=args.dataset_name, dataset_base_dir=args.data_base_dir
        )

        if args.dataset_type == "total":
            if args.dataset_name in ["popqa","nq","triviaqa"] and "total" in loaded_data_map:
                dataset = loaded_data_map["total"]
            else:
                for type_key in loaded_data_map:
                    dataset.extend(loaded_data_map[type_key])
                if not dataset and loaded_data_map:
                     print(f"Warning: Aggregated 'total' for {args.dataset_name} is empty. Classified types found: {list(loaded_data_map.keys())}")
        elif args.dataset_type in loaded_data_map:
            dataset = loaded_data_map[args.dataset_type]
        else:
            print(f"Error: Dataset type '{args.dataset_type}' not found for '{args.dataset_name}'. Available: {list(loaded_data_map.keys())}")
            return

        if not dataset:
            print(f"Error: No data loaded for {args.dataset_name}, type '{args.dataset_type}'.")
            return

        if args.start_question is not None and args.end_question is not None:
            dataset = dataset[args.start_question:args.end_question]
            print(f"Processing a subset of {len(dataset)} questions for {args.dataset_name} ({args.dataset_type}).")

        print(f"Successfully loaded {len(dataset)} questions for {args.dataset_name} (selected type: {args.dataset_type}).")

    except FileNotFoundError as e:
        print(f"Data loading error (FileNotFound): {e}.")
        return
    except KeyError as e:
        print(f"Data loading error (KeyError): {e}.")
        return
    except Exception as e:
        print(f"Unexpected data loading error: {e}")
        import traceback; traceback.print_exc()
        return

    print("\nInitializing models...")
    try:
        answer_generator = AnswerGenerator(args.model_name_or_path, args.device, args.base_seed)
        llm_judge = LLMJudge(args.judge_model_name_or_path, args.judge_device, current_run_judge_cache_file)

        if any(m in ['mqag', 'bertscore', 'ngram', 'nli', 'llm_prompt'] for m in args.detect_methods):
            try: selfcheck_wrapper = SelfCheckGPTWrapper(device=args.device)
            except Exception as e: print(f"Warning: Failed to init SelfCheckGPTWrapper: {e}")

        if 'semantic_entropy' in args.detect_methods:
            try: se_calculator = SemanticEntropyCalculator(nli_model_name_or_path="microsoft/deberta-large-mnli", device=args.device)
            except Exception as e: print(f"Warning: Failed to init SemanticEntropyCalculator: {e}")

        if 'seu' in args.detect_methods:
            try: seu_calculator = SemanticEmbeddingUncertaintyCalculator(device=args.device)
            except Exception as e: print(f"Warning: Failed to init SEUCalculator: {e}")

        if 'sindex' in args.detect_methods:
            try: sindex_calculator = SIndexCalculator(model_name_or_path=args.sindex_embedding_model, device=args.device, clustering_threshold=args.sindex_threshold)
            except Exception as e: print(f"Warning: Failed to init SIndexCalculator: {e}")

        if 'mind' in args.detect_methods and args.mind_classifier_path:
            try: mind_wrapper = MINDWrapper(classifier_model_path=args.mind_classifier_path, device=args.device)
            except Exception as e: print(f"Warning: Failed to init MINDWrapper: {e}")

        if 'saplma' in args.detect_methods and args.saplma_probe_path:
            try: saplma_wrapper = SAPLMAWrapper(probe_model_path=args.saplma_probe_path, device=args.device)
            except Exception as e: print(f"Warning: Failed to init SAPLMAWrapper: {e}")

        if 'eubhd' in args.detect_methods:
            try: eubhd_calculator = EUBHDCalculator(
                    device=args.device, idf_path=args.eubhd_idf_path, gamma=args.eubhd_gamma,
                    only_keyword=(not args.eubhd_all_tokens), use_penalty=(not args.eubhd_no_penalty),
                    use_idf=(not args.eubhd_no_idf))
            except Exception as e: print(f"Warning: Failed to init EUBHDCalculator: {e}")
    except Exception as e:
        print(f"Fatal error during model initialization: {e}")
        import traceback; traceback.print_exc()
        if 'llm_judge' not in locals(): llm_judge = None
        return

    setup_end_time = time.time()
    print(f"Model and Wrapper setup complete in {setup_end_time - script_start_time:.2f} seconds.")
    print(f"Methods to be executed: {args.detect_methods}")

    print("\n--- Starting Pipeline Execution ---")
    results_list = run_pipeline(
        args,
        dataset,
        answer_generator,
        llm_judge,
        current_run_judge_cache_file,
        selfcheck_wrapper,
        se_calculator,
        seu_calculator,
        sindex_calculator,
        mind_wrapper,
        saplma_wrapper,
        eubhd_calculator
    )

    # --- Save Final Results and Cache ---
    print("\n--- Saving final results and judge cache ---")
    if llm_judge is not None and hasattr(llm_judge, 'save_cache'):
        try:
            llm_judge.save_cache()
            print(f"Final judge cache for this run saved to: {current_run_judge_cache_file}")
        except Exception as e:
            print(f"Error during final save of judge cache: {e}")
    elif llm_judge is None:
         print("LLM Judge was not initialized, skipping judge cache save.")

    # Save results (JSONL)
    results_filename = f"results_s{args.base_seed}"
    if args.start_question is not None and args.end_question is not None:
        results_filename += f"_q{args.start_question}_{args.end_question}"
    results_filename += ".jsonl"
    output_path_for_results = os.path.join(current_run_output_dir, results_filename)
    save_results_jsonl(results_list, output_path_for_results)

    # --- Cleanup Models ---
    print("\nCleaning up models (final)...")
    del answer_generator
    del llm_judge
    if 'selfcheck_wrapper' in locals() and selfcheck_wrapper is not None: del selfcheck_wrapper
    if 'se_calculator' in locals() and se_calculator is not None: 
        try:
            se_calculator.cleanup()
        except Exception as se_cleanup_e:
            print(f"Error during SE cleanup: {se_cleanup_e}")
        del se_calculator
    if 'seu_calculator' in locals() and seu_calculator is not None: del seu_calculator
    if 'sindex_calculator' in locals() and sindex_calculator is not None: del sindex_calculator
    if 'mind_wrapper' in locals() and mind_wrapper is not None: del mind_wrapper
    if 'saplma_wrapper' in locals() and saplma_wrapper is not None: del saplma_wrapper
    if 'eubhd_calculator' in locals() and eubhd_calculator is not None: del eubhd_calculator
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Model cleanup finished.")

    script_end_time = time.time()
    print(f"\nExperiment finished. Total time: {script_end_time - script_start_time:.2f} seconds.")
    print(f"Results for this run are in: {output_path_for_results}")
    if os.path.exists(current_run_judge_cache_file):
        print(f"Judge cache for this run is at: {current_run_judge_cache_file}")
    else:
        print(f"Judge cache for this run (intended path): {current_run_judge_cache_file} (File not found - check if judgments were made and cache saving was successful).")

if __name__ == "__main__":
    main()
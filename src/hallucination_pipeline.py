import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import gc
from typing import Dict, List, Any, Tuple, Optional

from src.detection_methods.hallucination_detection.semantic_entropy import SemanticEntropyCalculator
from src.detection_methods.hallucination_detection.seu_calculator import SemanticEmbeddingUncertaintyCalculator
from src.detection_methods.hallucination_detection.sindex_calculator import SIndexCalculator
from src.detection_methods.hallucination_detection.mind_wrapper import MINDWrapper
from src.detection_methods.hallucination_detection.saplma_wrapper import SAPLMAWrapper
from src.detection_methods.hallucination_detection.eubhd_calculator import EUBHDCalculator
from src.answer_generator import AnswerGenerator
from src.llm_judge import LLMJudge
from src.detection_methods.hallucination_detection.selfcheckgpt_wrapper import SelfCheckGPTWrapper

INF_REPLACEMENT_VALUE = 1e9

PP_PE_METRIC_KEYS = ["lnpp", "lnpe"]

SELFCHECK_SCORE_KEYS_MAP = {
    "mqag": "mqag",
    "bertscore": "bertscore",
    "nli": "nli",
    "ngram": "ngram_unigram_max",
}

METHODS_REQUIRING_SAMPLES_BASE = {'mqag', 'bertscore', 'ngram', 'nli', 'ptrue', 'semantic_entropy', 'seu', 'sindex'}
METHODS_REQUIRING_LOGPROBS_BASE = {'semantic_entropy'}
METHODS_REQUIRING_TARGET_MODEL = {'mind', 'saplma', 'eubhd'}

def run_pipeline(args: argparse.Namespace,
                 dataset: List[Dict[str, Any]],
                 answer_generator: AnswerGenerator,
                 llm_judge: LLMJudge,
                 judge_cache_file_path: Optional[str],
                 selfcheck_wrapper: Optional[SelfCheckGPTWrapper],
                 se_calculator: Optional[SemanticEntropyCalculator],
                 seu_calculator: Optional[SemanticEmbeddingUncertaintyCalculator],
                 sindex_calculator: Optional[SIndexCalculator],
                 mind_wrapper: Optional[MINDWrapper],
                 saplma_wrapper: Optional[SAPLMAWrapper],
                 eubhd_calculator: Optional[EUBHDCalculator]
                 ) -> List[Dict[str, Any]]:
    """
    Runs the main generation, judging, and detection pipeline.
    Returns a list of dictionaries containing results for each question.
    """
    results_list = []
    processed_count = 0
    error_count = 0

    methods_requiring_samples_list = [
        m for m in args.detect_methods if m in METHODS_REQUIRING_SAMPLES_BASE
    ]
    methods_requiring_logprobs_list = [
        m for m in args.detect_methods if m in METHODS_REQUIRING_LOGPROBS_BASE
    ]
    
    for question_data in tqdm(dataset, desc="Processing Questions"):
        qid = question_data.get('qid', f'index_{processed_count}')
        processed_count += 1
        item_result = {}

        try:
            # 1. Generate Answers and Probabilities/Metrics
            generated_results = answer_generator.generate(
                question_data,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                main_do_sample=True,
                main_temperature=args.main_temp,
                main_top_p=args.main_top_p,
                sample_temperature=args.sample_temp,
                sample_top_p=args.sample_top_p,
                methods_require_samples=methods_requiring_samples_list,
                methods_require_logprobs=methods_requiring_logprobs_list
            )
            main_answer = generated_results.get('main_answer', f"Error: Generation failed QID {qid}")
            sample_answers = generated_results.get('sample_answers', [])
            pp_pe_metrics = generated_results.get('pp_pe_metrics', {})
            sequence_log_probs = generated_results.get('sequence_log_probs', [])

            item_result.update({
                "qid": qid,
                "question": question_data.get('question', ''),
                "main_answer": main_answer,
                "sample_answers": sample_answers,
            })

            # 2. Judge Main Answer
            if main_answer and not main_answer.startswith("Error:"):
                try:
                    judge_verdict = llm_judge.judge(
                        question_data,
                        main_answer,
                        max_retries=args.judge_max_retries,
                        retry_delay=args.judge_retry_delay
                    )
                    item_result["judge_verdict"] = judge_verdict
                except Exception as judge_e:
                    item_result["judge_verdict"] = f"judging_error: {judge_e}"
            else:
                item_result["judge_verdict"] = "judgment_skipped_generation_error"

            # 3. Run Detection Methods
            detection_scores = {}
            samples_available = bool(sample_answers)
            can_run_sample_methods = main_answer and not main_answer.startswith("Error:") and samples_available
            valid_samples = []
            if can_run_sample_methods:
                valid_samples = [s for s in sample_answers if isinstance(s, str) and s.strip() and not s.startswith("Error:")]
                if not valid_samples and len(sample_answers) > 0:
                    can_run_sample_methods = False

            # PP/PE
            for key in PP_PE_METRIC_KEYS:
                if key in args.detect_methods:
                    score_value = pp_pe_metrics.get(key)
                    if score_value is not None:
                        if np.isinf(score_value): 
                            score_value = INF_REPLACEMENT_VALUE if score_value > 0 else -INF_REPLACEMENT_VALUE
                        elif np.isnan(score_value): 
                            score_value = None
                        else: 
                            score_value = float(score_value)
                    detection_scores[key] = score_value

            # Run SelfCheckGPT methods
            if can_run_sample_methods and selfcheck_wrapper:
                scg_methods = [m for m in args.detect_methods if m in SELFCHECK_SCORE_KEYS_MAP]
                if scg_methods:
                    try:
                        scg_scores = selfcheck_wrapper.detect(
                            main_answer=main_answer,
                            sample_answers=valid_samples,
                            question=question_data.get('question'),
                            methods_to_run=scg_methods,
                            split_sentences=args.split_sentences_for_detection
                        )
                        for method in scg_methods:
                            score_key = SELFCHECK_SCORE_KEYS_MAP.get(method)
                            if score_key in scg_scores:
                                detection_scores[method] = scg_scores[score_key]
                    except Exception as scg_e:
                        print(f"\nError during SelfCheck detection for QID {qid}: {scg_e}")

            # Run other detection methods
            if can_run_sample_methods:
                # PTrue
                if 'ptrue' in args.detect_methods:
                    try:
                        ptrue_prob = answer_generator.calculate_ptrue_probability(
                            question=question_data['question'],
                            main_answer=main_answer,
                            sample_answers=valid_samples,
                            num_samples_in_prompt=args.num_samples
                        )
                        if ptrue_prob is not None:
                            detection_scores['ptrue'] = 1.0 - ptrue_prob
                    except Exception as ptrue_e:
                        print(f"\nError during PTrue calculation for QID {qid}: {ptrue_e}")

                # Semantic Entropy
                if 'semantic_entropy' in args.detect_methods and se_calculator:
                    try:
                        se_score = se_calculator.calculate_score(
                            question=question_data['question'],
                            answers=[main_answer] + valid_samples,
                            sequence_log_probs=sequence_log_probs
                        )
                        detection_scores['semantic_entropy'] = se_score
                    except Exception as se_e:
                        print(f"\nError during Semantic Entropy calculation for QID {qid}: {se_e}")

                # SEU
                if 'seu' in args.detect_methods and seu_calculator:
                    try:
                        seu_score = seu_calculator.calculate_score(
                            main_answer=main_answer,
                            sample_answers=valid_samples
                        )
                        detection_scores['seu'] = seu_score
                    except Exception as seu_e:
                        print(f"\nError during SEU calculation for QID {qid}: {seu_e}")

                # SIndex
                if 'sindex' in args.detect_methods and sindex_calculator:
                    try:
                        sindex_score = sindex_calculator.calculate_score(
                            question=question_data['question'],
                            main_answer=main_answer,
                            sample_answers=valid_samples
                        )
                        detection_scores['sindex'] = sindex_score
                    except Exception as sindex_e:
                        print(f"\nError during SIndex calculation for QID {qid}: {sindex_e}")

            # Run model-dependent methods
            if main_answer and not main_answer.startswith("Error:"):
                # MIND
                if 'mind' in args.detect_methods and mind_wrapper:
                    try:
                        mind_score = mind_wrapper.calculate_score(
                            question=question_data['question'],
                            llm_answer=main_answer,
                            target_model=answer_generator.model,
                            target_tokenizer=answer_generator.tokenizer
                        )
                        detection_scores['mind'] = mind_score
                    except Exception as mind_e:
                        print(f"\nError during MIND calculation for QID {qid}: {mind_e}")

                # SAPLMA
                if 'saplma' in args.detect_methods and saplma_wrapper:
                    try:
                        saplma_score = saplma_wrapper.calculate_score(
                            question=question_data['question'],
                            llm_answer=main_answer,
                            target_model=answer_generator.model,
                            target_tokenizer=answer_generator.tokenizer,
                            layer_index=args.saplma_layer
                        )
                        detection_scores['saplma'] = saplma_score
                    except Exception as saplma_e:
                        print(f"\nError during SAPLMA calculation for QID {qid}: {saplma_e}")

                # EUBHD
                if 'eubhd' in args.detect_methods and eubhd_calculator:
                    try:
                        eubhd_score = eubhd_calculator.calculate_score(
                            question=question_data['question'],
                            llm_answer=main_answer,
                            target_model=answer_generator.model,
                            target_tokenizer=answer_generator.tokenizer
                        )
                        detection_scores['eubhd'] = eubhd_score
                    except Exception as eubhd_e:
                        print(f"\nError during EUBHD calculation for QID {qid}: {eubhd_e}")

            item_result["detection_scores"] = detection_scores

        except Exception as loop_e:
            print(f"\nFATAL Error processing item QID {qid}: {loop_e}")
            import traceback; traceback.print_exc()
            error_count += 1
            item_result = {
                "qid": qid,
                "question": question_data.get('question', 'N/A'),
                "error_message": f"Loop failed - {loop_e}",
                "main_answer": "Error: Loop failed",
                "sample_answers": [],
                "judge_verdict": "Error: Loop failed",
                "detection_scores": {}
            }

        results_list.append(item_result)

        if processed_count % 50 == 0 and judge_cache_file_path and llm_judge:
            print(f"\nSaving judge cache (at item {processed_count})...")
            try:
                llm_judge.save_cache()
            except Exception as cache_e:
                print(f"Error saving judge cache: {cache_e}")

    return results_list
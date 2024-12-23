from datetime import datetime 
import sys
import argparse
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
from evaluation_utils import (
    load_json,
    execute_sql,
    package_sqls,
    sort_results,
    print_data,
)


def result_callback(result):
    exec_result.append(result)


def calculate_ex(predicted_res, ground_truth_res):
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(
    predicted_sql, ground_truth, db_place, idx, meta_time_out, sql_dialect
):
    try:
        res = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth, db_place, sql_dialect, calculate_ex),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f"timeout",)]
        res = 0
    except Exception as e:
        result = [(f"error",)]  # possibly len(query) > 512 or not executable
        res = 0
    result = {"sql_idx": idx, "res": res}
    return result


def run_sqls_parallel(
    sqls, db_places, num_cpus=1, meta_time_out=30.0, sql_dialect="SQLite"
):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):

        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(
                predicted_sql,
                ground_truth,
                db_places[i],
                i,
                meta_time_out,
                sql_dialect,
            ),
            callback=result_callback,
        )
    pool.close()
    pool.join()
def find_res_of_matched_index(exec_results, content)->int:
    content_index = content["question_id"]
    for i, res in enumerate(exec_results):
        if res["sql_idx"] == content_index:
            return res
    raise ValueError(f"Index {content_index} not found in the results.")


def compute_acc_by_diff(exec_results, diff_json_path):
    num_queries = len(exec_results)
    results = [res["res"] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    for i, content in enumerate(contents):
        if content["difficulty"] == "simple":
            try:
                simple_results.append(find_res_of_matched_index(exec_results,content))
            except:
                pass

        if content["difficulty"] == "moderate":            
            try:
                moderate_results.append(find_res_of_matched_index(exec_results,content))
            except:
                pass

        if content["difficulty"] == "challenging":
            try:
                challenging_results.append(find_res_of_matched_index(exec_results,content))
            except:
                pass

    simple_acc = sum([res["res"] for res in simple_results]) / len(simple_results)
    moderate_acc = sum([res["res"] for res in moderate_results]) / len(moderate_results)
    challenging_acc = sum([res["res"] for res in challenging_results]) / len(
        challenging_results
    )
    all_acc = sum(results) / num_queries
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        num_queries,
    ]
    return (
        simple_acc * 100,
        moderate_acc * 100,
        challenging_acc * 100,
        all_acc * 100,
        count_lists,
    )


if __name__ == "__main__":
    # Record the start time
    start_time = datetime.now()
    print(f"Start time: {start_time}")

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--predicted_sql_path", type=str, required=True, default=""
    )
    args_parser.add_argument("--ground_truth_path", type=str, required=True, default="")
    args_parser.add_argument("--data_mode", type=str, required=True, default="dev")
    args_parser.add_argument("--db_root_path", type=str, required=True, default="")
    args_parser.add_argument("--num_cpus", type=int, default=1)
    args_parser.add_argument("--meta_time_out", type=float, default=30.0)
    args_parser.add_argument("--mode_gt", type=str, default="gt")
    args_parser.add_argument("--mode_predict", type=str, default="gpt")
    args_parser.add_argument("--difficulty", type=str, default="simple")
    args_parser.add_argument("--diff_json_path", type=str, default="")
    args_parser.add_argument("--engine", type=str, default="")
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths, index_pred = package_sqls(
        args.predicted_sql_path,
        args.db_root_path,
        args.engine,
        sql_dialect=args.sql_dialect,
        mode=args.mode_predict,
        data_mode=args.data_mode,
    )
    # generate ground truth sqls:
    gt_queries, db_paths_gt, index_gt = package_sqls(
        args.ground_truth_path,
        args.db_root_path,
        args.engine,
        sql_dialect=args.sql_dialect,
        mode="gt",
        data_mode=args.data_mode,
    )

    gt_queries = gt_queries[1135:]
    db_paths_gt = db_paths_gt[1135:]
    index_gt = index_gt[1135:]

    query_pairs = []
    db_paths_new = []
    index_of_pred = 0
    for idx in range(len(gt_queries)):
        if str(idx + 1135) in index_pred:
            query_pairs.append((pred_queries[index_of_pred], gt_queries[idx]))
            db_paths_new.append(db_paths[index_of_pred])
            index_of_pred += 1
        else:
            db_paths_new.append("")
            query_pairs.append(("", gt_queries[idx]))

    run_sqls_parallel(
        query_pairs,
        db_places=db_paths_new,
        num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
        sql_dialect=args.sql_dialect,
    )
    exec_result = sort_results(exec_result)
    print("start calculate")
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(
        exec_result, args.diff_json_path
    )
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print(f"EX for {args.engine} on {args.sql_dialect} set")
    print("start calculate")
    print_data(score_lists, count_lists, metric="EX")
    print(
        "==========================================================================================="
    )
    print(f"Finished EX evaluation for {args.engine} on {args.sql_dialect} set")
    print("\n\n")


    # Record the end time and calculate duration
    end_time = datetime.now()
    print(f"End time: {end_time}")
    duration = end_time - start_time
    print(f"Duration: {duration}")

    # Saving EX results in a txt file
    ex_result_file_path = args.predicted_sql_path + "ex_score.txt"
    ex_file_content = f"The EX scores are: \nOverall: {acc} \nSimple: {simple_acc} \nModerate: {moderate_acc} \nChallenging: {challenging_acc} \n\nCount Lists: {count_lists} \n\nEvaluation Duration: {duration} \n\nMeta Time-out: {args.meta_time_out}"
    with open(ex_result_file_path, 'w') as f:
        f.write(ex_file_content)

    print("EX score is written into the ex_score.txt file.")

    





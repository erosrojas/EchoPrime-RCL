import os
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

from utils.video_utils import process_video_path

def get_report_vid_pairs(df, tokenizer, batch_size, verbose: bool = False):

    # Group by exam_id and get the first report per exam
    grouped = df.groupby("exam_id")["extracted_findings"].first().reset_index()

    report_dict = {}

    exam_ids = grouped["exam_id"].tolist()
    reports = grouped["extracted_findings"].tolist()

    for i in tqdm(range(0, len(reports), batch_size), desc="Checking report token length"):
        batch_reports = reports[i:i+batch_size]
        batch_exam_ids = exam_ids[i:i+batch_size]

        tokens = tokenizer(batch_reports, padding=False, truncation=False, add_special_tokens=False)

        for j, input_ids in enumerate(tokens["input_ids"]):
            if len(input_ids) <= 512:
                report_dict[batch_exam_ids[j]] = batch_reports[j]


    # Build vid_dict using filtered exam_ids
    vid_dict = {}
    for exam in tqdm(report_dict.keys()):
        vid_paths = df[df["exam_id"] == exam]["processed_file_address"].tolist()
        vid_dict[exam] = vid_paths

    bad_vids_count = 0
    good_vids_count = 0
    min_number_batches = 0
    # Group video paths by report
    report_to_vids = {}
    for exam, vids in tqdm(vid_dict.items(), desc="Checking if video paths are valid"):

        good_vids = []
        for vid in vids:
            vid_path = f'/data{vid[15:]}'

            if os.path.exists(vid_path):
                good_vids.append(vid_path)
                good_vids_count += 1
            else:
                bad_vids_count += 1

        if good_vids:
            report = report_dict[exam]
            report_to_vids[report] = good_vids
            
        min_number_batches = max(min_number_batches, len(good_vids))

    num_batches = max(math.ceil(good_vids_count/batch_size), min_number_batches)

    batches = [[] for _ in range(num_batches)]
    batch_lengths = [0] * num_batches

    i = 0
    for report, vids in tqdm(report_to_vids.items(), desc="Sorting video-report pairs into batches"):
        for vid in vids:
            attempts = 0
            found_batch = False
            while attempts < num_batches:
                batch_idx = i % num_batches
                if batch_lengths[batch_idx] < batch_size:
                    batches[batch_idx].append((report, vid))
                    batch_lengths[batch_idx] += 1
                    found_batch = True
                    break
                i += 1
                attempts += 1
            if not found_batch:
                raise ValueError("Unable to find batch for video")
            i += 1  # Move to next batch for next video of same report

    batch_sizes = [len(batch) for batch in batches]

    if verbose:
        print(batch_sizes)
        print(f"Number of missing videos: {bad_vids_count}")

    # Flatten batches into final list
    report_vid_pairs = [pair for batch in batches for pair in batch]

    return report_vid_pairs, batch_sizes

def batch_generator(df, tokenizer, batch_size):
    report_vid_pairs, batch_sizes = get_report_vid_pairs(df, tokenizer, batch_size)

    idx = 0
    for batch_size in batch_sizes:
        batch = report_vid_pairs[idx:idx + batch_size]
        idx += batch_size

        # Process this specific batch
        reports = [r for r, _ in batch]
        paths = [v for _, v in batch]

        padded = tokenizer(reports, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        input_ids = padded["input_ids"]
        attention_mask = padded["attention_mask"]

        with ThreadPoolExecutor() as executor:
            videos = list(executor.map(process_video_path, paths))
        video_tensor = torch.stack(videos)

        yield video_tensor, input_ids, attention_mask
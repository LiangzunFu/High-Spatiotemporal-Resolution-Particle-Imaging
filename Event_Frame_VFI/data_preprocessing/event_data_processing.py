# 将事件相机拍摄到的raw文件根据trigger信号拆分为与帧相机每一帧对应的npz文件

import os
import numpy as np
from metavision_core.event_io import EventsIterator

# raw_file = r"D:\mCloudDownload\12.raw"
# output_folder = r"D:\dataset\Frame_Event_Dataset\dataset_12\Events1"

def split_events_by_triggers(raw_file, output_folder, delta_t_us=1000000):
    os.makedirs(output_folder, exist_ok=True)

    # 创建 iterator
    mv_iterator = EventsIterator(
        raw_file,
        mode="delta_t",
        start_ts=0,
        delta_t=delta_t_us,
        relative_timestamps=False
    )

    all_events = None
    all_triggers = None

    # 遍历事件流，收集 trigger 上升沿
    for evs in mv_iterator:
        if evs.size != 0:
            triggers = mv_iterator.reader.get_ext_trigger_events()
            if len(triggers) > 0:

                # 去重：基于 (x, y, t, p)
                packed = (evs['x'].astype(np.uint64) << 48) | \
                         (evs['y'].astype(np.uint64) << 32) | \
                         (evs['t'].astype(np.uint64) << 1) | \
                         (evs['p'].astype(np.uint64))

                # 唯一化并取索引
                _, unique_idx = np.unique(packed, return_index=True)
                evs = evs[unique_idx]
                # 去重结束

                if all_events is None:
                    # 第一次拿到事件
                    all_events = evs.copy()
                    all_triggers = triggers.copy()
                else:
                    all_events = np.concatenate((all_events, evs))
                    all_triggers = np.concatenate((all_triggers, triggers))
                mv_iterator.reader.clear_ext_trigger_events()

    # 筛选 trigger 上升沿，并保留前两项 (去掉无用 id 信息)
    all_triggers = [(trigger[0], trigger[1]) for trigger in all_triggers if trigger[0] == 1]

    if len(all_triggers) == 0:
        print("No triggers found.")
        return

    # 提取 trigger 时间戳
    trigger_times = [trigger[1] for trigger in all_triggers]

    duration = []
    # 遍历 trigger，保存每两个 trigger 之间的事件
    for i in range(len(trigger_times)):
        t_start = trigger_times[i]
        if i < len(trigger_times) - 1:
            t_end = trigger_times[i + 1]
            duration.append(t_end - t_start)
        else:
            t_end = trigger_times[i] + int(sum(duration) / len(trigger_times))  # 最后一个 trigger

        mask = (all_events['t'] >= t_start) & (all_events['t'] < t_end)
        segment = all_events[mask]

        if segment.size == 0:
            print(f"Segment {i}: no events between {t_start} - {t_end}, saving empty EventSlice_{i}.npz.")
            out_path = os.path.join(output_folder, f"EventSlice_{i}.npz")
            np.savez_compressed(out_path,
                                x=np.array([], dtype=np.uint16),
                                y=np.array([], dtype=np.uint16),
                                t=np.array([], dtype=np.uint64),
                                p=np.array([], dtype=np.uint16),
                                t_start = t_start,
                                t_end = t_end
            )
            continue

        segment = segment[np.argsort(segment['t'])]
        out_path = os.path.join(output_folder, f"EventSlice_{i}.npz")
        np.savez_compressed(out_path,
                            x=segment['x'],
                            y=segment['y'],
                            t=segment['t'],
                            p=segment['p'],
                            t_start = t_start,
                            t_end = t_end
        )
        print(f"Saved {len(segment)} events to {out_path}")

    print("Done splitting events by trigger signals.")

# split_events_by_triggers(raw_file, output_folder, delta_t_us=1000000)

for i in range(5,29):
    raw_file = rf"D:\mCloudDownload\{i}.raw"
    if os.path.exists(raw_file):
        output_folder = rf"D:\dataset\Frame_Event_Dataset\dataset_{i}\Events1"
        split_events_by_triggers(raw_file, output_folder, delta_t_us=1000000)
# -*- coding: utf-8 -*-
        pd.DataFrame(columns=[
            "symbol", "name", "date", "window", "box_lower", "box_upper",
            "box_width_pct", "inbox_ratio_pct", "touch_upper_count",
            "touch_lower_count", "below_lower_count", "prev_close", "last_close",
            "breakout_pct_vs_upper", "box_max_close", "score", "pattern_note"
        ]).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print("\n没有筛到符合条件的股票。")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    summary_lines = [
        f"batch_index={BATCH_INDEX}",
        f"batch_total={BATCH_TOTAL}",
        f"stocks_scanned={len(universe)}",
        f"matched={len(matched)}",
        f"failed={len(failed)}",
        f"elapsed_seconds={elapsed:.2f}",
        f"elapsed_hms={hours}小时 {minutes}分钟 {seconds:.2f}秒",
    ]
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    if len(universe) > 0:
        print(f"平均每只股票耗时: {elapsed / len(universe):.2f} 秒")


if __name__ == "__main__":
    main()
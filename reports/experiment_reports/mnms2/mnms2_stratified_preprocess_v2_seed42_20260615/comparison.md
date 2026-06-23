# M&Ms-2 Fair Comparison

| case | threshold | postprocess | dice | iou | hd | hd95 | assd | precision | recall | topology_success_rate | topology_failure_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_no_post | 0.5800 | none | 0.8547 | 0.7641 | 3.3330 | 2.2580 | 0.7768 | 0.8529 | 0.8670 | 0.8617 | 170 |
| baseline_post | 0.5800 | closing_r1_lcc_fill_extra_holes_preserve_largest_hole | 0.8548 | 0.7643 | 3.3328 | 2.2605 | 0.7772 | 0.8522 | 0.8680 | 0.8617 | 170 |
| r2net_no_post | 0.7800 | none | 0.8572 | 0.7683 | 3.4055 | 2.3021 | 0.7638 | 0.8735 | 0.8515 | 0.8560 | 177 |
| r2net_post | 0.7800 | closing_r1_lcc_fill_extra_holes_preserve_largest_hole | 0.8576 | 0.7687 | 3.4027 | 2.3024 | 0.7630 | 0.8731 | 0.8525 | 0.8625 | 169 |

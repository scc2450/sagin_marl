# Line 3 Episode Compare Table

Source files:

- [debug_ep03_seed43000base.csv](d:/研三上/毕设/sagin_marl/runs/line3_short/setpool30/debug_exports/debug_ep03_seed43000base.csv)
- [debug_ep13_seed43000base.csv](d:/研三上/毕设/sagin_marl/runs/line3_short/setpool30/debug_exports/debug_ep13_seed43000base.csv)
- [debug_ep04_seed43000base.csv](d:/研三上/毕设/sagin_marl/runs/line3_short/setpool30_initcritic/debug_exports/debug_ep04_seed43000base.csv)

Notes:

- `closing_speed > 0` means the current most dangerous pair is still closing.
- `closing_speed < 0` means the pair is separating.
- `correction` here is the maximum correction norm among the three UAVs at that step.

| episode | pair | step | min_dist | closing_speed | correction | queue_total_active |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ep03 | 1-2 | 378 | 275.246 | 35.699 | 0.000 | 0.000 |
| ep03 | 1-2 | 379 | 238.279 | 36.962 | 0.000 | 0.000 |
| ep03 | 1-2 | 380 | 200.065 | 38.205 | 0.000 | 0.000 |
| ep03 | 1-2 | 381 | 160.619 | 39.432 | 0.000 | 0.000 |
| ep03 | 1-2 | 382 | 119.967 | 40.619 | 0.000 | 0.000 |
| ep03 | 1-2 | 383 | 78.185 | 41.674 | 0.000 | 0.000 |
| ep03 | 1-2 | 384 | 35.713 | 41.603 | 0.000 | 0.000 |
| ep03 | 1-2 | 385 | 13.562 | -29.587 | 0.857 | 0.000 |
| ep13 | 0-2 | 350 | 204.717 | 39.256 | 0.000 | 0.000 |
| ep13 | 0-2 | 351 | 165.818 | 44.191 | 0.000 | 0.000 |
| ep13 | 0-2 | 352 | 122.406 | 41.608 | 0.000 | 0.000 |
| ep13 | 0-2 | 353 | 83.912 | 33.182 | 0.000 | 0.000 |
| ep13 | 0-2 | 354 | 61.727 | 6.582 | 0.000 | 0.000 |
| ep13 | 0-2 | 355 | 73.670 | -27.687 | 0.000 | 0.000 |
| ep13 | 0-2 | 356 | 109.649 | -41.499 | 0.000 | 0.000 |
| ep13 | 0-2 | 357 | 150.146 | -34.697 | 0.000 | 0.000 |
| ep13 | 0-2 | 358 | 187.175 | -38.678 | 0.000 | 0.000 |
| ep04 | 0-1 | 343 | 430.497 | 60.256 | 0.000 | 0.000 |
| ep04 | 0-1 | 344 | 370.005 | 60.457 | 0.000 | 0.000 |
| ep04 | 0-1 | 345 | 309.306 | 60.662 | 0.000 | 0.000 |
| ep04 | 0-1 | 346 | 245.298 | 64.084 | 0.000 | 0.000 |
| ep04 | 0-1 | 347 | 181.397 | 63.353 | 0.000 | 0.000 |
| ep04 | 0-1 | 348 | 119.161 | 60.302 | 0.000 | 0.000 |
| ep04 | 0-1 | 349 | 62.235 | 62.678 | 0.000 | 0.000 |
| ep04 | 0-1 | 350 | 2.588 | -3.668 | 0.000 | 0.000 |

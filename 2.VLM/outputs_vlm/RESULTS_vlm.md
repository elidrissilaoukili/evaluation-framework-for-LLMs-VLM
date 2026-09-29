## VLM results (BLIP-base captioning, BLIP-VQA, CLIP retrieval)

Captioning, 200 Flickr30k images: BLEU-4 0.247, ROUGE-L 0.476, METEOR 0.369, CIDEr 0.513, CLIPScore(L/14) 0.583

RL metrics: J(pi)=0.2544 [0.2502, 0.2585], SCST advantage=-0.0316 [-0.0367, -0.0266], hybrid=0.4263 [0.4068, 0.4457]

### Retrieval
| model | i2t_R@1 | i2t_R@5 | i2t_R@10 | i2t_medR | t2i_R@1 | t2i_R@5 | t2i_R@10 | t2i_medR | rsum |
|---|---|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 76.90 | 95.90 | 98.30 | 1.00 | 59.76 | 84.42 | 90.78 | 1.00 | 506.06 |
| CLIP ViT-L/14 | 84.50 | 97.50 | 99.20 | 1.00 | 66.22 | 88.56 | 92.90 | 1.00 | 528.88 |

### Best-of-k / reward-hacking check
| metric | greedy | best-of-k by reward | delta |
|---|---|---|---|
| CLIP-B/32 cos (optimised reward) | 0.2860 | 0.3054 | 0.0193 |
| CLIP-L/14 cos (independent judge) | 0.2333 | 0.2467 | 0.0134 |
| METEOR (human refs) | 0.3686 | 0.3282 | -0.0405 |
| ROUGE-L (human refs) | 0.4765 | 0.4100 | -0.0665 |
| CIDEr (human refs) | 0.5128 | 0.4423 | -0.0705 |
| avg caption length (words) | 6.7000 | 7.8150 | 1.1150 |

### VQAv2
acc 0.847 [0.808, 0.883], J(pi) 0.711 [0.675, 0.750], SCST adv -0.136 [-0.168, -0.103]

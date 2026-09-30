# Secondary analyses (post hoc). Primary metric is the uncapped corpus WER; see DESIGN.md 'Secondary analyses'.

runaway = I >= 20 (by-I) or hyp words > 1.5 * ref words + 10 (by-len). capped = runaway hyp cut to ref words + 5 (uses the reference). dur-capped = every hyp cut to ceil(4.5 * audio s) + 5 words (deployable); truncated (n) = hyps shortened; refs truncated = references the same cap would cut (must be 0).

| file | utts | WER | S | D | I | runaway | by-I | by-len | WER excl. runaway | WER capped | WER dur-capped | truncated (n) | refs truncated | empty hyps |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final/A1-fp32-finetune-test-clean.json | 2620 | 5.41% | 2171 | 276 | 422 | 0 | 0 | 0 | 5.41% | 5.41% | 5.41% | 0 | 0 | 0 |
| final/A2-ternary-test-clean.json | 2620 | 50.46% | 18802 | 3070 | 4887 | 2 | 2 | 0 | 50.36% | 50.43% | 50.46% | 0 | 0 | 0 |
| final/A3-ternary-embed-test-clean.json | 2620 | 76.31% | 30479 | 2827 | 7161 | 2 | 2 | 1 | 76.19% | 76.23% | 76.30% | 1 | 0 | 0 |
| final/A1-fp32-finetune-test-other.json | 2939 | 12.95% | 5258 | 668 | 920 | 0 | 0 | 0 | 12.95% | 12.95% | 12.95% | 0 | 0 | 0 |
| final/A2-ternary-test-other.json | 2939 | 71.86% | 28748 | 3740 | 5512 | 1 | 1 | 0 | 71.82% | 71.84% | 71.86% | 0 | 0 | 0 |
| final/A3-ternary-embed-test-other.json | 2939 | 87.58% | 36506 | 3258 | 6552 | 5 | 5 | 2 | 87.38% | 87.44% | 87.58% | 1 | 0 | 0 |

## Worst runaway utterances by insertions (max 3 per file)

**final/A1-fp32-finetune-test-clean.json: 0 runaway utterances hold 0 of 422 insertions**


**final/A2-ternary-test-clean.json: 2 runaway utterances hold 40 of 4887 insertions**

- `2300-131720-0004  S 33 D 0 I 20  ref 44 w, hyp 64 w`
  - ref: owing to his insistence on low pressure direct current for use in densely populated districts as the only safe and truly
  - hyp: only to his system is law will not worry or direct direct for you said then sleep i will be able to dig this asked the only safe and truly you are a little frock for a little way up to a little later like a lank or 2 and a genteelers at a c
- `4077-13754-0011  S 22 D 0 I 20  ref 33 w, hyp 53 w`
  - ref: federal judges and united states attorneys in utah who were not mormons nor lovers of mormonism refused to entertain com
  - hyp: that your own treasures and idea is they to carry these into top who are not more miss the lower source of women as a free nature to obtain them place or profit or keep use of the law because it is a man of interest in justinian and that it

**final/A3-ternary-embed-test-clean.json: 2 runaway utterances hold 71 of 7161 insertions**

- `4077-13754-0002  S 34 D 0 I 50  ref 45 w, hyp 95 w`
  - ref: it was through floyd is advice that buchanan ordered the military expedition to utah ostensibly to install certain feder
  - hyp: he was forced to put his advice he had given the words of a very good natie and he taught them also to sleep in search of all their fellows who had failed to reproach him as he had managed to tell him that they had a chide with him that it 
- `8463-294828-0038  S 23 D 0 I 21  ref 35 w, hyp 56 w`
  - ref: 1000s of handkerchiefs were waving above these tightly packed masses hailing the abraham lincoln until it reached the wa
  - hyp: that is another height of which we were being of about these times to pay back to massies and i know you may bring a way and a great reach to the water is side of the river and to take the tip of the long capricians at the far end of the ye

**final/A1-fp32-finetune-test-other.json: 0 runaway utterances hold 0 of 920 insertions**


**final/A2-ternary-test-other.json: 1 runaway utterances hold 20 of 5512 insertions**

- `7105-2330-0041  S 22 D 0 I 20  ref 28 w, hyp 48 w`
  - ref: the local trade unionists took offense at the fact of cabinet ministers having personally acted as strike breakers and e
  - hyp: though the oak trees you need not ask to look at venus i would have to have got a ministrations having been a pretty i would take at last tried grace and even in their ease of the better bath they ate a basket of advice i then

**final/A3-ternary-embed-test-other.json: 5 runaway utterances hold 119 of 6552 insertions**

- `6938-70848-0019  S 20 D 0 I 28  ref 26 w, hyp 54 w`
  - ref: on the 27th occurred the debate on the land question which revealed the differences between the agrarian program of the 
  - hyp: on the time he said i would have a good to be on there and i wish you can not be good for me if it has been for good and good and good of you and i will go over and see them and that is the matter of course she is lost
- `367-293981-0010  S 66 D 2 I 25  ref 88 w, hyp 111 w`
  - ref: sancho got up with pain enough in his bones and went after the innkeeper in the dark and meeting the officer who was loo
  - hyp: so to have got with the opinion of his bill and it would have been to the end of the darkeningings of his son who was to see what he had said of him he said to himself that he would have remarked to him do you think it is a great pity that 
- `6070-86744-0022  S 43 D 0 I 24  ref 51 w, hyp 75 w`
  - ref: talking of countries replied franz of what country is the count what is his native tongue whence does he derive his imme
  - hyp: till he knew much to his wife is friend so much to tell what he chose there was to take his aid to do what was his he did not send his fortune i am brought up the old man is said he is a very lad as i know is it not a matter of his time he 


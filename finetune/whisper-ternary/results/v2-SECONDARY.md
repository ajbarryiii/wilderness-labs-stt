# Secondary analyses (post hoc). Primary metric is the uncapped corpus WER; see DESIGN.md 'Secondary analyses'.

runaway = I >= 20 (by-I) or hyp words > 1.5 * ref words + 10 (by-len). capped = runaway hyp cut to ref words + 5 (uses the reference). dur-capped = every hyp cut to ceil(4.5 * audio s) + 5 words (deployable); truncated (n) = hyps shortened; refs truncated = references the same cap would cut (must be 0).

| file | utts | WER | S | D | I | runaway | by-I | by-len | WER excl. runaway | WER capped | WER dur-capped | truncated (n) | refs truncated | empty hyps |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final/v2-A0-fp32-zeroshot-test-clean.json | 2620 | 5.66% | 2217 | 379 | 403 | 0 | 0 | 0 | 5.66% | 5.66% | 5.66% | 0 | 0 | 0 |
| final/v2-A1-fp32-finetune-test-clean.json | 2620 | 4.50% | 1780 | 258 | 349 | 0 | 0 | 0 | 4.50% | 4.50% | 4.50% | 0 | 0 | 0 |
| final/v2-A2-ternary-test-clean.json | 2620 | 13.12% | 4351 | 1014 | 1590 | 6 | 6 | 2 | 12.35% | 12.47% | 12.84% | 2 | 0 | 0 |
| final/v2-A3-ternary-embed-test-clean.json | 2620 | 12.12% | 4048 | 1210 | 1171 | 8 | 8 | 1 | 11.65% | 11.82% | 12.11% | 1 | 0 | 0 |
| final/v2-A0-fp32-zeroshot-test-other.json | 2939 | 14.54% | 5392 | 775 | 1523 | 2 | 2 | 2 | 13.70% | 13.75% | 13.77% | 2 | 0 | 0 |
| final/v2-A1-fp32-finetune-test-other.json | 2939 | 12.37% | 4817 | 621 | 1103 | 1 | 1 | 1 | 12.16% | 12.18% | 12.20% | 2 | 0 | 0 |
| final/v2-A2-ternary-test-other.json | 2939 | 30.90% | 11278 | 1496 | 3566 | 8 | 8 | 6 | 29.37% | 29.55% | 30.12% | 5 | 0 | 0 |
| final/v2-A3-ternary-embed-test-other.json | 2939 | 28.42% | 10589 | 1698 | 2744 | 8 | 7 | 3 | 27.74% | 27.85% | 28.13% | 2 | 0 | 0 |

## Worst runaway utterances by insertions (max 3 per file)

**final/v2-A0-fp32-zeroshot-test-clean.json: 0 runaway utterances hold 0 of 403 insertions**


**final/v2-A1-fp32-finetune-test-clean.json: 0 runaway utterances hold 0 of 349 insertions**


**final/v2-A2-ternary-test-clean.json: 6 runaway utterances hold 375 of 1590 insertions**

- `908-157963-0008  S 9 D 1 I 177  ref 43 w, hyp 219 w`
  - ref: thou gentle maid of silent valleys and of modest brooks for thou shall be clothed in light and fed with morning manna ti
  - hyp: thou gentle made of silent valleys and of modest brooks for thou shall be clothed in the light and fed with mourning man till summers heat melts thee beside the fountains and the springs deflors in eternal veil they thy thy thy thy thy thy 
- `2094-142345-0008  S 34 D 0 I 82  ref 88 w, hyp 170 w`
  - ref: but there is always a stronger sense of life when the sun is brilliant after rain and now he is pouring down his beams a
  - hyp: but there is always a stronger sense of life when the sun is brilliant after rain and now he is pouring down his beams and making sparkles among the wet straw and lighting up every patch of vivid green moss on the red tiles of the cow shed 
- `1284-134647-0005  S 8 D 0 I 38  ref 61 w, hyp 99 w`
  - ref: they asserted with confidence and almost with exultation that the apostolical succession was interrupted that all the bi
  - hyp: they asserted with confidence and almost with exultation that the apostolical succession was interrupted that all the bishops of the europe and asia were infected by the contagion of guilt and schism and that the prerogatives of the catholi

**final/v2-A3-ternary-embed-test-clean.json: 8 runaway utterances hold 217 of 1171 insertions**

- `5639-40744-0027  S 17 D 1 I 53  ref 64 w, hyp 116 w`
  - ref: thus saying and pressing the crucifix to her breast she fell fainting into the arms of dona estafania who as a gentlewom
  - hyp: thus saying pressing the crucifix to her breast she fell fainting into the arms of donna is to fanny who has a gentle woman to whose sex pity is a natural as cruelty as to a man instantly pressed her lips to those of the fainting girl shedd
- `908-31957-0015  S 4 D 1 I 27  ref 41 w, hyp 67 w`
  - ref: that was the chrism of love which love is own crown with sanctifying sweetness did precede the 3rd upon my lips was fold
  - hyp: that was the chrysum of love which loves own crown with sanctifying sweetness did proceed the 3rd upon my lips was folded down in perfect purple state since when indeed i have been proud and said my love my love my love my love my love my l
- `4507-16021-0032  S 16 D 1 I 26  ref 62 w, hyp 87 w`
  - ref: he must descend with his heart full of charity and severity at the same time as a brother and as a judge to those impene
  - hyp: he must descend with his heart full of charity and severity at the same time as a brother and as a judge those impenetrable casemates where crawl hell now those who bleed in those who will both those who will both who will both who will bot

**final/v2-A0-fp32-zeroshot-test-other.json: 2 runaway utterances hold 430 of 1523 insertions**

- `3528-168669-0049  S 7 D 0 I 218  ref 7 w, hyp 225 w`
  - ref: fauchelevent held his peace she went on
  - hyp: the fact that he was a little bit more than a little bit more than a little bit more than a little bit more than a little bit more than a little bit more than a little bit more than a little bit more than a little bit more than a little bit
- `3538-163622-0007  S 10 D 0 I 212  ref 13 w, hyp 225 w`
  - ref: come hither come hither my handsome son and let me comb your hair
  - hyp: come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and come in and 

**final/v2-A1-fp32-finetune-test-other.json: 1 runaway utterances hold 104 of 1103 insertions**

- `4852-28312-0016  S 6 D 0 I 104  ref 10 w, hyp 114 w`
  - ref: would that interfere with jakey is getting the job sir
  - hyp: would that interfere with jay ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki ki 

**final/v2-A2-ternary-test-other.json: 8 runaway utterances hold 766 of 3566 insertions**

- `3528-168669-0082  S 13 D 0 I 172  ref 33 w, hyp 205 w`
  - ref: his order has produced 40 popes 200 cardinals 50 patriarchs 1600 archbishops 4600 bishops 4 emperors 12 empresses 46 kin
  - hyp: his order has produced 40 post 200 cardinals 50 petre x 1600 archbishops 4600 bishops 4 emperors 12 empresses 46 kings 41 queens ¢3680 dollars dollars dollars dollars dollars dollars dollars dollars dollars dollars dollars dollars dollars d
- `3764-168671-0009  S 29 D 0 I 158  ref 61 w, hyp 219 w`
  - ref: the interment of mother crucifixion in the vault under the altar the exit of cosette the introduction of jean valjean to
  - hyp: the internality of mother crucifixion in the vault under the altar the accident of cosette the introduction of joval joined into the dead dream will i be executed without difficulty and the head be known hatch that is remarked in passing th
- `3331-159609-0014  S 41 D 0 I 140  ref 80 w, hyp 220 w`
  - ref: it was so tender earnest and defiant that fanny forgot the defense of her own lover in admiration of polly is loyalty to
  - hyp: it was so tender earnest and defiant that fanny forgot the defense of her own love and admiration of party is fire to her for this faithful or of a throbbing laugh send you a revelation to fanny who was seized to hearing her friends burst o

**final/v2-A3-ternary-embed-test-other.json: 8 runaway utterances hold 360 of 2744 insertions**

- `2033-164914-0021  S 9 D 5 I 175  ref 52 w, hyp 222 w`
  - ref: we will do thee no upright 0 my son nor wrong thee in aught but our object is that thou bend thy gracious steps with me 
  - hyp: we will do thee no upright only son nor wrong thee inaud our object is that thou band thy gracious steps with me to my mistress receive her answer and returning wheel and safety and thou shalt have a handsome presence as one who is one who 
- `3538-163624-0023  S 41 D 0 I 46  ref 42 w, hyp 88 w`
  - ref: then brynhild is father told gunnar that she would marry none but him who could ride the flame in front of her enchanted
  - hyp: whimbrin n n n ern ern er n ern b er n ern ern ern n n n er n er n er in er n r n r n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n
- `8461-281231-0024  S 8 D 0 I 33  ref 58 w, hyp 91 w`
  - ref: he left the gallant band of foresters sorrowing deeply for his lost friend the lord of coningsburgh and he and his follo
  - hyp: he left the gun and band of forest thus sorrowing deeply for his lost friend the lord of conningsburg and he and his followers had scast departed when a procession moved slowly from under the green wood branches in the direction which he ha


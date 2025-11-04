# Part 2
Det egna projektet, steg 2
I denna del av projektet ska du implementera och utvärdera en maskininlärningslösning
baserad på den analys du genomförde i steg 1. Målet är att bygga en fungerande modell
som kan göra förutsägelser utifrån ditt valda dataset.
Steg 1: Förberedelse av data
- [X] Ladda in och undersök ditt dataset i en Jupyter Notebook eller valfri miljö.
2. Rengör datat genom att:
- [X]  Hantera saknade värden.
- [~]  Normalisera/standardisera numeriska data vid behov.
- [ ]  Omvandla kategoriska variabler till numeriska representationer om det
behövs.
- [-] Eventuellt skapa nya features som kan förbättra modellen.

Steg 2: Träna en maskininlärningsmodell
- [X] Välj en lämplig modell (exempelvis beslutsträd, linjär regression, neuralt nätverk
etc.).
- [X] Dela upp datasetet i träning- och testdata.
- [X] Träna modellen och optimera hyperparametrar vid behov.
Steg 3: Utvärdera modellen
- [X] Testa modellen på testdatan.
  - Check the different runs.

● En kort rapport (ca 1-2 sidor) som beskriver:
○ Hur datat förberetts.
The data has ben moved into datasets of different sizes for the first few runs. This was done for checking the mode quickly.
The full data was used at first, to see how accurate the model would become.
- [ ] TODO: Double check The labels were changed to Numerical values, A-Z -> 0->25 or 1->26
- [ ] Clean the data to reduce variance


○ Vilken modell som valts och varför.
I first used a SVG model which does linear regression. It was almost as bad as ranomly assigning labels - Around 5% compared to randomly assigning 1/26th.
The assumption was that the linear regression model would perform badly on the categorization of the data since the data is not linear.
I tested a few different models, SDG Classifier with and without regression and Decision Tree. From the data I found that the Decision tree was more accurate and decided to experiment with it.
- [ ] TODO: Add Model tested with Regression,
- [ ] TODO: doublecheck every name
I created a test to improve the model using the parameters 'split' and 'leaves'
- [ ] TODO: Add link to split and leaves parameters.
After testing I saw that increasin the 'split' and 'leaves' did increase the accuracy, with an increase cost in time.
I decided to see if the model would perform better if the images size was reduced and tried 32x32 (original), 16x16 ad 8x8.
- [ ] TODO: Add data from test with differen image size.

I want to check if the datasets variance is the reason for the accuracry.
- [ ] TODO: Clean data so that the dataset has lower variance. To test 'garbage in, garbage out'

- [ ] TODO: Show results from the runs in a nice table


2. Utvärdera resultatet med relevanta metoder, exempelvis:
- [X] Klassificeringsproblem: Accuracy, precision, recall, F1-score.

- [ ] Analysera och reflektera över modellens prestanda.
- [ ] Om prestandan är låg, föreslå förbättringar och motivera varför de skulle fungera.
Inlämning
○ Hur modellen presterade och eventuella förbättringsförslag.
- [ ] TODO: Add time for the mode and describe how it performs. Learning and on test data.


● Jupyter Notebook eller kodfiler som visar hela arbetet.
- [ ] TODO: Add gihub code

Betygskriterier
För G:
● En fungerande modell har implementerats.
● Datasetet har hanterats och anpassats för träning.
● Utvärdering har genomförts med relevanta metoder.
● En kort rapport som beskriver processen och resultaten.
För VG:
● Modellen har optimerats och förbättrats genom hyperparametertuning eller feature
engineering.
● Analysen innehåller en djupare diskussion om modellens prestanda och möjliga
förbättringar.
● Alternativa metoder/tester har utförts för att jämföra modellens effektivitet.
● Koden är välkommenterad och strukturerad på ett tydligt sätt.
20251026_SDGClassifier_full_run_with_code.md
Accuracry: 0.68389
20251026_DecisionTree_full_run_with_code.md
Accuracry: 0.82238

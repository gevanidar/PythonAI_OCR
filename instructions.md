# Part 1
Det egna projektet, steg 1
Du skall nu skapa ett eget projekt, där du använder maskininlärning/AI till att lösa ett på förhand
givet problem.
Du väljer själv vilket problem du vill försöka lösa. Här finns inga egentliga begränsningar.
Exempel på uppgifter är:
• Identifiera saker/djur/människor i bilder
• Tolka handskriven text
• Förutsäga väder
• Förutsäga börskurser
• Tolka patientdata för att förutsäga risk för vissa sjukdomar
• Identifiera ”fake news”
Börja med att i text beskriva problemet och vad det är du vill åstadkomma. Nästa steg är att försöka
hitta data som kan hjälpa dig lösa ditt problem. Om du inte kommer över lämpligt data får du tänka
om. Kan du göra en variant på ditt problem som kan lösas med det data du hittat? Om inte, definiera
ett nytt problem och försök igen.
Tänk på att datat kan komma från olika källor. Om du hämtar data från olika källor måste du
undersöka hur kompatibelt det är.
Lär känna ditt data.
• Är det komplett?
• Har du null-värden?
• Har du extrema värden?
• Vilka datatyper har datat?
• Vilka fält i ditt data vill du använda dig av?
• Hur kan du konvertera alla fält du vill använda till ett numeriskt format?
Typ av problem
Vilken typ av probem är det du försöker lösa? Är det ett klassificerings- eller regressionsproblem? Är
ditt data labeled (alltså har du färdiga svar att mata din modell med vid träning). Om inte, hur skall
du få det? Skall du själv skapa labels eller vill du försöka att ta hjälp av unsupervised learning
(fungerar bra på klassificeringsproblem men inte regressionsproblem).
Vill du i stället arbeta med Reinforcement Learning och deep neural networks? Du får du fundera vad
du vill uppnå och vilket data du vill mata ditt nätverk med. Hur skall det vara strukturerat? Hur skall
ditt belönings/bestraffningssystem se ut. Hur skall du tolka det output du får från nätverket? Vilken
träningsplattform vill du använda för att träna dit närverk?
Dokumentera
Detta är det du skall göra i steg 1. Du skall alltså inte börja implementera någon lösning ännu, utan
undersöka förutsättningarna för ditt projekt. Skapa ett dokument där du beskriver problemet och
den lösning (enligt frågeställningarna ovan) som du kommit fram till. Om du använder dataset,
beskriv det data du hittat. Hur många rader har du? Hur är kvaliteten på datat.
Du kommer lämna in detta dokument i nästa inlämning. Du kan också, om du vill, lämna in Jupyter
Notebooks som visar hur du har undersökt ditt data (om du inte skall arbeta med Reinforcement
Learning).

# Part 2
Det egna projektet, steg 2
I denna del av projektet ska du implementera och utvärdera en maskininlärningslösning
baserad på den analys du genomförde i steg 1. Målet är att bygga en fungerande modell
som kan göra förutsägelser utifrån ditt valda dataset.
Steg 1: Förberedelse av data
1. Ladda in och undersök ditt dataset i en Jupyter Notebook eller valfri miljö.
2. Rengör datat genom att:
○ Hantera saknade värden.
○ Normalisera/standardisera numeriska data vid behov.
○ Omvandla kategoriska variabler till numeriska representationer om det
behövs.
○ Eventuellt skapa nya features som kan förbättra modellen.
Steg 2: Träna en maskininlärningsmodell
1. Välj en lämplig modell (exempelvis beslutsträd, linjär regression, neuralt nätverk
etc.).
2. Dela upp datasetet i träning- och testdata.
3. Träna modellen och optimera hyperparametrar vid behov.
Steg 3: Utvärdera modellen
1. Testa modellen på testdatan.
2. Utvärdera resultatet med relevanta metoder, exempelvis:
○ Klassificeringsproblem: Accuracy, precision, recall, F1-score.
○ Regressionsproblem: MSE, RMSE, R^2.
3. Analysera och reflektera över modellens prestanda.
4. Om prestandan är låg, föreslå förbättringar och motivera varför de skulle fungera.
Inlämning
● En kort rapport (ca 1-2 sidor) som beskriver:
○ Hur datat förberetts.
○ Vilken modell som valts och varför.
○ Hur modellen presterade och eventuella förbättringsförslag.
● Jupyter Notebook eller kodfiler som visar hela arbetet.
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

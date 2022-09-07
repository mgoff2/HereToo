## Title: Kaggle: Feedback Prize - Predicting Effective Arguments: An Instructive Post-Mortem*

**Goal: Rate the effectiveness of argumentative writing elements from students grade 6-12**

**Host: Georgia State University**

**Final Submission Deadline: 23 August 2022**

**Total Submissions Made: 7**

**Outcome: Log-Loss = .689 | Position = 1004/1557**

Goal Details: To 'classify argumentative elements in student writing as "effective," "adequate," or "ineffective." You will create a model trained on data that is representative of the 6th-12th grade population in the United States in order to minimize bias. Models derived from this competition will help pave the way for students to receive enhanced feedback on their argumentative writing. With automated guidance, students can complete more assignments and ultimately become more confident, proficient writers.'

Data: The dataset presented here contains argumentative essays written by U.S students in grades 6-12. These essays were annotated by expert raters for discourse elements commonly found in argumentative writing: Lead - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis Position - an opinion or conclusion on the main question Claim - a claim that supports the position Counterclaim - a claim that refutes another claim or gives an opposing reason to the position Rebuttal - a claim that refutes a counterclaim Evidence - ideas or examples that support claims, counterclaims, or rebuttals. Concluding Statement - a concluding statement that restates the claims Your task is to predict the quality rating of each discourse element. Human readers rated each rhetorical or argumentative element, in order of increasing quality, as one of: Ineffective, Adequate, Effective.

Topics: Text Cleaning, Tokenization, Lemmatization, Sentiment Analysis, Feature Engineering, Classification Algorithms, GridSearch, Transformers, LSTM

Modules: NLTK, Gensim, Transformers, Spacy, Sklearn, Textstat, Plotly, Matplotlib, Seaborn, Keras, Pytorch, Tensorflow

(*Eliminated subtitles include: 'A Smorgasbord of Tentative Panache', 'Lessons in Coding and Overconfidence', and 'An Exhaustive/Exhausting Treatment')

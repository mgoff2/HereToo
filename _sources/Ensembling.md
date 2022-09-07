### Ensembles

There are multiple ways I experimented with ensembling (and, to cover my bases, I know that some of the above models are technically 'ensembles' in themselves, but I mean the term here to suggest two or more essentially disconnected models that are manually combined, rather than a single pre-defined algorithm). ((Note that here I don't execute any of the following programs, I just include them for illustrative purposes. They, like the GridSearch, can get pretty time-consumptive.))
- Voting: 
The simplest way to ensemble is using sklearn's VotingClassifier in conjunction with models of different strengths and unrelated algorithms (i.e., distance-based with decision-based, for example). Multiple models are iteratively trained-and-tested, then at the time of prediction, they can either vote 'hard' - each cast a predicted class label, and most represented class wins - or 'soft' - each submits a probability score, and the mean of those probabilities returns the predicted class based on highest likelihood. (Respectively, these are 'classification voting' and 'regression voting').
- Stacking: 
Having base models predict, then having a meta-model take the predictions from the other learners as its dataset. Easy to implement with SKlearn's Stacking Classifier. 
- Modified Stacking:
One approach I tried was simply to append the outputs of a strong (but not the strongest) model to the dataframe, then use that as a feature for the strongest learner. Specifically, since there was no model that was both performing well *and* capturing a significant portion of the ineffective group, I wanted to find one that performed well on ineffectives, then see if the strongest model could pick up on that hint. 
- Boosting 
Boosting is, of course, used above, so I won't include a demo here. The simple version is that the model iteratively applies 'weak learners' that are essentially random guesses to define boundary regions. After N iterations, however, the final result is to classify the correctly grouped items in the majority of learners as being members of whatever that set happened to be. (Read more here/Tip o' the Hat to Jocelyn D'Souza: [Link.](https://medium.com/greyatom/a-quick-guide-to-boosting-in-ml-acf7c1585cb5))
- Blending
AKA 'stacked generalization,' this technique uses a very similar approach to stacking, but instead of having the meta-model predict on a dataset of the base-models' predictions of the *folds*, it has the meta-model predict on base models' predictions on a held-out testing portion. Presumably this is too prevent data-leakage. I didn't implement it. 
- Bagging 
A portmanteau of 'Bootstrap' and 'Aggregating,' it means sub-sampling of the data is used *with replacement* -- aka 'bootstrapping.' Instead of the 'folds' consisting of distinct items, any given fold could have observation x in the context of p, or q, or r, thus it's reappearance forces the model to evaluate it's weight in terms of *all* of the models. Think of it as interviewing an employee at the office-- then at the ballgame, then the bar, then with friends, then with family... the different environments evince different features and ideas. 
- Rule-Based Classification 
In a rule-based classification (sometimes called a 'hybrid' model), the analyst specifies a set of conditions under which the model's predictions are modified or overruled. There are number of scenarios in which this might be valuable. Sparsity is one of them: since probabilistic models struggle with underrepresented features or classes, compelling the machine to look for and consider them in a distinctive way can help with identification. Likewise, when boundaries are unclear in a statistical sense, but obvious in a conceptual sense, rules can help. For example, consider all of the metrics computed in this notebook, but being used to compare Hemingway and Shakespeare. Obviously, both should be classified as 'Effective' writers, but things like punctuation frequency, sentence length, diction, and so forth will be profoundly different between them. Simply forcing the final model with a rule 'If Author_ID = "Hemingway": Effectiveness = 'Effective' can help iron out incongruences in a model that otherwise works well. 
Unfortunately, no scientific module that I was able to find makes incorporating user-defined rules into a model architecture particularly easy. While object-oriented and/or functional programming can be used to tweak any extant model to your preferences, I found it a little overwhelming and didn't pursue it for this project. I've added the link below, though, to my 'Learn-Soon' bookmark folder, and am eager to explore it further. 
(Tip o' the Hat to Lukas Haas [Link.](https://towardsdatascience.com/hybrid-rule-based-machine-learning-with-scikit-learn-9cb9841bebf2))
- Manually Weighted Combinations
Finally, the most common ensemble technique I saw used in other Kaggle Notebooks was a manual weight assignment. When you go to make a submission, the requirements specify that you pass the 'test.csv' file given in the competition data to your model, then put those predictions into a .csv file in the format shown below.

```python
#Example of simple voting classifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
svm2 = LinearSVC(max_iter =10000)

xgb3 = XGBClassifier(objective = 'multi:softmax', random_state = 42, num_class = 3)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[('svm2', svm2),
                ('xgb3', xgb3)],
                voting='hard')

TotalPipe = Pipeline([('features', features),
                   ('clf', voting_clf)
                   ])
```

```python
#Stacking Classifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Define the standard learners you want to estimate from the dataframe
Dataframe_models = [
                 ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
                 ('rf_2', KNeighborsClassifier(n_neighbors=5))             
                ]
#Define the 'final_estimator' using StackingClassifer (here=Logistic Regression);
#Bonus: the module has a built in cross_validation which is applied to the entire dataframe models, to avoid overfitting.
Meta_model = StackingClassifier(estimators=base_learners,
                         final_estimator=LogisticRegression(),  
                         cv=10)
```

```python
#Example of 'Modified Stacking'
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

x = KFP_df.model_tokens
y = KFP_df.discourse_effectiveness
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.3, random_state = 42)

x_train_tf = vec_tfidf.fit_transform(x_train)
nbc = MultinomialNB()
nbc.fit(x_train_tf, y_train)

xvectors = vec_tfidf.transform(x)
y_hat = nbc.predict(xvectors)

y_hatz = pd.Series(y_hat)
y_hatz['nbc_preds'] = y_hat

KFP_df = KFP_df.merge(y_hatz.to_frame(), how = 'left', left_index = True, right_index = True)
KFP_df.rename(columns = { 0 : 'nbc_preds'}, inplace = True)
KFP_df.nbc_preds = KFP_df.nbc_preds.astype('int64')
```

```python
#Bagging
from sklearn.ensemble import BaggingClassifier

vec_tfidf = TfidfVectorizer(ngram_range = (1,1), 
                            sublinear_tf = True,
                            analyzer = 'word', 
                            norm = 'l2',
                            max_df = 20000,
                            min_df = 1,                              
                            encoding = 'latin-1',
                            stop_words = stop_words)


x = np.array(dfcopy.discourse_text)
X = vec_tfidf.fit_transform(x)
y = KFP_df.discourse_effectiveness
x_train, x_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)

log_reg = LogisticRegression(random_state = 0, max_iter = 1000)

bgclf = BaggingClassifier(base_estimator=log_reg, n_estimators=100,
                                 max_features=10,
                                 max_samples=100,
                                 random_state=1, n_jobs=5)

bgclf.fit(x_train, y_train)

print('Training Score: %.3f, ' %bgclf.score(x_train, y_train),
      'Model training Score: %.3f' %bgclf.score(x_test, y_test))
```

```python
#Manual Weight Assignment 
predsF = []
predsF.append(fullPipe.predict_proba(xF))
predsF = np.array(predsF).mean(0)

submission = pd.read_csv("sample_submission.csv")
submission['Ineffective'] = predsF[:,0]
submission['Adequate'] = predsF[:,1]
submission['Effective'] = predsF[:,2]
submission.to_csv("./submission.csv", index=False)
submission.head()
```

```python
from tabulate import tabulate
head = ['discourse_id', 'Ineffective', 'Adequate', 'Effective']
data = [['0', 'a261b6e14276', 0.348259, 0.603059 ,0.048682],
['1', '5a88900e7dc1', 0.611058, 0.305413, 0.083528],
['2', '9790d835736b', 0.626507, 0.229162, 0.144330],
['3', '75ce6d68b67b', 0.625603, 0.184476, 0.189921],
['4', '93578d946723', 0.558659, 0.287924, 0.153416]]

print(tabulate(data, headers = head, tablefmt = 'grid'))

+----+----------------+---------------+------------+-------------+
|    | discourse_id   |   Ineffective |   Adequate |   Effective |
+====+================+===============+============+=============+
|  0 | a261b6e14276   |      0.348259 |   0.603059 |    0.048682 |
+----+----------------+---------------+------------+-------------+
|  1 | 5a88900e7dc1   |      0.611058 |   0.305413 |    0.083528 |
+----+----------------+---------------+------------+-------------+
|  2 | 9790d835736b   |      0.626507 |   0.229162 |    0.14433  |
+----+----------------+---------------+------------+-------------+
|  3 | 75ce6d68b67b   |      0.625603 |   0.184476 |    0.189921 |
+----+----------------+---------------+------------+-------------+
|  4 | 93578d946723   |      0.558659 |   0.287924 |    0.153416 |
+----+----------------+---------------+------------+-------------+
```

To manually weight the submission file, you could just generate a *set* of files like these, one per model, then specify the operation how you wanted it. Whereas the ensembling techniques above use averages or votes, you can tailor this type of combination to your preferences. For example, if you had a logistic regression model's output that you felt strongly about, a DeBERTa model that you liked but didn't love, and a Naive Bayes classifier that you felt had limited utility, you could write take the three outputs, and run:

```python
Final_Submission_Scores = (logreg * 0.5) + (DebertaModel * 0.4) + (NBClf * 0.1)
```


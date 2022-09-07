#!/usr/bin/env python
# coding: utf-8

# # Part 1: Understanding the Competition

# A point of central importance is that this competition is *not* about mimicking or capturing what's going on with, say, 
# Grammarly. The rubric is very open-ended and includes nothing whatever about 1) punctuation, 2) spelling, 3) grammatical formality/correctness, 4) style, or 5) diction. From the description: 
# 
# > There are numerous automated writing feedback tools currently available, but they all have limitations, especially with argumentative writing. Existing tools often fail to evaluate the quality of argumentative elements, such as organization, evidence, and idea development.
# 
# This makes sense to me: grammarly, spellcheck, and so on can already assist with all of these, and the probability of a student having mastered these skills is something we can (sadly) already predict through data on race, income, zipcode, home-value, and so on. Also, as a former teacher, I can attest to the severity of the headache induced by grading thousands of documents a year. A model that could learn a rubric-- whatever it consisted of-- and then grade accordingly would be immensely valuable to our educators (and the immediacy of feedback to our students). Indeed, this seems to be the ambition of the project, as it states on the competition home page: 
# 
# >An automated feedback tool is one way to make it easier for teachers to grade writing tasks assigned to their students that will also improve their writing skills.
# 
# It goes on to specify that in addition to the things accompolished by existing writing feedback tools, they hope to develop a tool with more complexity. Indeed, the rubric (linked below) details almost philosophical criteria like 'validity,' 'effectiveness,' direction of attention, 'stance-taking,' 'clarity,' 'relevance,' 'acceptability,' 'objectivity', 'soundness,' 'substantiation,' and restatement. I personally think that the task of getting AI to recognize *logic* and *valid reasoning* is an extremely useful, lofty, and interesting goal. In an information-sphere awash with deep-fakes, conspiracy theories, false-leads, counterintelligence, propaganda, and general misinformation, an algorithm that could discern between sound and unsound reasoning could be of real use, whether in education or industry. 
# 
# [Rubric](https://docs.google.com/document/d/1G51Ulb0i-nKCRQSs4p4ujauy4wjAJOae/edit)
# 
# >The dataset presented here contains argumentative essays written by U.S students in grades 6-12. These essays were annotated by expert raters for discourse elements commonly found in argumentative writing:
# >- Lead - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis
# >- Position - an opinion or conclusion on the main question
# >- Claim - a claim that supports the position
# >- Counterclaim - a claim that refutes another claim or gives an opposing reason to the position
# >- Rebuttal - a claim that refutes a counterclaim
# >- Evidence - ideas or examples that support claims, counterclaims, or rebuttals.
# >- Concluding Statement - a concluding statement that restates the claims
# 
# In plain-language, in other words, does the argument make any sense? Imagine that the writer were instead speaking his or her argument (allowing for us to largely ignore grammar, spelling, interjections, and the like. Does the speaker stay on topic? Does the speaker state a position clearly? Does the speaker drive-home the point? Does the speaker show that A follows from B, and that factual evidence demonstrates B is true? Does the speaker consider opposing points of view, and if so, are they treated as men of straw or of steel? Assume the writer has an editor, or is using grammarly, or is just spit-balling a first-draft which will later be revised into a presentable document. Does the core structure and point come across and have any merit?
# 
# By taking a look at the rubric above, we can get a sense of exactly what the graders are looking for. The rubric consists of the types of discourse-elements, what they are characterized by, what makes them effective or ineffective, and an illustrative example-element: 
# 
# <div class="alert alert-block alert-info">
# Argumentation Element: Lead
#     
# >Rating = Effective: <br>
# Discourse Prompt: “Should we admire heroes but not celebrities?<br>
# Description: The lead grabs the reader’s attention and strongly points toward the position.<br>
# Example: ' Too often in today's society people appreciate only the circumstances which immediately profit themselves and they follow the pack, never stopping to wonder, "Is this really the person who deserves my attention?" '<br>
# 
# >Rating = Adequate: <br>
# Prompt: “Can people ever be truly original?”<br>
# Description: The lead attempts to grab the reader’s attention and points toward the position.<br>
# Example: 'Originality: being able to do something that few or none have done before.'<br>
# 
# >Rating = Ineffective<br>
# Prompt: “Can people ever be truly original?”<br>
# Description: The lead may not grab the readers' attention and may not point to the position.<br>
# Example: 'Originality is hard to in this time and era.'<br>
# </div>
# 
# Sometimes, the examples given are quite long. For instance, when describing an effective 'Evidence' element, they offer the following: 
# 
# <div class="alert alert-block alert-info"> 
#     
# >Rating = Effective <br>
#     
# >Description: The evidence is closely relevant to the claim they support and back up the claim objectively with concrete facts, examples, research, statistics, or studies. The reasons in the evidence support the claim and are sound and well substantiated.<br>
# 
# >(Rather than a prompt, the 'context' is a previous claim made by the student): Claim: "There are a number of instances, in either our everyday lives or special occurrences, in which one must confront an issue they do not want to face." <br>
# 
# >Example: "For instance, the presidential debate is currently going on and in order to choose the right candidate, they must research all sides of both candidates. The voter must learn all about the morals and how each one plans to better America. This might disturb some people, given that some people may either feel too strongly about a certain candidate or that they may not feel strongly enough. However, by not researching and gaining all the possible knowledge that they can, they are hurting themselves by passing up a valuable opportunity to possibly better the country for themselves and the people surrounding them." <br>
# </div>
# 
# Ostensibly, then, the task is to 1) to hunt through a collection of examples of this type; 2) to assess them in light of these descriptions of efficacy, 3) and then predict algorithmically whether or not the text either a) contains content consistent with these descriptions or (more sublty) b) was *assessed by a human rater to have satisifed the description.* (More later.)
# 
# Some immediate associations I have are:
# - *Integrative complexity*: a term used in psychology to describe something like 'tolerance for cognitive dissonance; ability to see two or more sides or positions; willingness to validate a legitimate point, even when it contradicts a previously held belief; serious consideration of evidence, as opposed to defensive dismissal or the issuing of rhetorical red-herrings.' 
# - *Sophistry*: the use of language in an ostensibly coherent and persuasive way, but which is actually misleading or illogical; robust rhetorically, but substantially specious.
# - *Default Bias*: a concept from behavioral economics describing a tendency to fall back on a simple heuristic of the kind, 'Unless (insert major factor) is present, I'll just do (insert some default behavior).' In this case, I suspect that teachers/graders are likely tired, bored, hurried, and otherwise ready to be done with this process, and thus will have some degree of a "Unless this essay *really* captures my attention (in a good or bad way), it's 'Adequate'."
# - *English Teacher Bias*: a term I just made up, but which I propose to mean the instinctive negative reaction experienced by bookish, scholastic types when they see misspellings, grammtically inchoate sentences, poor punctuation, and so on. The rules don't specify, I know, that these things are *supposed* to factor into the grading, but that doesn't mean it doesn't. 
# - General Human Bias: Researchers in psychology and microeconomics have found a startling array of 'predictably irrational' behavior (tip o' the hat to Dan Ariely) which humans are subject to, and often they are only tenuously linked to the topic of study. For example, judges in criminal courts have been found to sentence more harshly on the basis of how tired they are, or how long it's been since they last ate. In general, I suspect that some of that same inconsistency will invariably be reflected in a dataset of this type. 

# ### 1: Understanding the Dataset

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import os

KFP_df = pd.read_csv("train.csv")

KFP_df.head(5)


# In[2]:


KFP_df.shape


# In[3]:


type(KFP_df)


# In[4]:


KFP_df.isnull().sum()


# In[5]:


KFP_df.describe()


# In[6]:


KFP_df.info()


# In[7]:


KFP_df.discourse_effectiveness.value_counts()


# In[8]:


testdict = dict(KFP_df['discourse_effectiveness'].value_counts())
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
myList = testdict.items()
myList = sorted(myList)
x,y = zip(*myList)
plt.bar(x, y)
plt.gcf().set_size_inches(6,4)
plt.show();


# In[9]:


testdict = dict(KFP_df['discourse_effectiveness'].value_counts())

discSect = KFP_df['discourse_type']
discEff = KFP_df['discourse_effectiveness']
newb = pd.concat([discSect, discEff], axis = 1)
newbCT = pd.crosstab(discSect, discEff, margins = True, margins_name="total")
newbCT


# In[10]:


newbCT.plot.bar()
plt.gcf().set_size_inches(8,5)
plt.show();


# In[11]:


newbCT2 = pd.crosstab(discSect, discEff, normalize = 'index')
newbCT2


# In[12]:


plot2 = newbCT2.plot.barh(stacked = True, color = ['gold', 'forestgreen','firebrick'])
plot2.set(xlabel= 'effectiveness_proportion', ylabel = 'discourse_type')
plot2.legend(bbox_to_anchor = (1.05, .6));


# ### Playing with Plotly (and TextStat)
# These two libraries make it almost embarassingly easy to generate interesting textual statistical summaries, as well as beautiful, interactive graphics. 
# (Tip o' the hat to Mr. Deepak Kaura for bringing this to my attention. [Link.](https://www.kaggle.com/code/deepakkaura/student-writing-the-visualization))

# In[13]:


import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Here, I'm applying a bit of cleaning to the text. These methods will be explored in detail later on, but I'm temporarily doing so here to prevent some of the visuals being overly influenced by meaningless fluff and messy data.
# I'll avoid modifying the original discourse_text, as some of the visuals will be more informative in their given state. 

# In[14]:


KFP_df['TextForVisuals'] = KFP_df['discourse_text'].astype(str).str.lower()
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
KFP_df['TextForVisuals'] = KFP_df['TextForVisuals'].apply(strip_punctuation)
KFP_df['TextForVisuals'] = KFP_df['TextForVisuals'].apply(remove_stopwords)
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
KFP_df['TextForVisuals'] = KFP_df['TextForVisuals'].apply(lambda x: tokenizer.tokenize(x))
KFP_df['TextForVisuals'] = KFP_df['TextForVisuals'].apply(lambda x: " ".join([i for i in x if len(i) > 1]))


# In[15]:


KFP_df['TFV_length'] = KFP_df['TextForVisuals'].apply(lambda x: len([i for i in x]))
KFP_df['TFV_length'].describe()


# Relatedly, we can see that there are some massive outliers on the high-end (texts up to ~20x the mean length). These will make the graphs look all wonky, so I'm going to push the edges in towards a more normal distribution.

# In[16]:


KFP_viz = KFP_df.copy(deep = True)
meanLength = KFP_df.TFV_length.mean()
stdLength = KFP_df.TFV_length.std()
maxLength = meanLength*3*stdLength
KFP_viz_norm = KFP_viz.loc[KFP_viz['TFV_length'] < maxLength]


# In[17]:


import textstat

KFP_viz_norm['Reading_Time'] = 0
KFP_viz['Average_Char_per_Word'] = 0
KFP_viz_norm['Reading_Ease'] = 0
KFP_df['Average_Sen_Length'] = 0
KFP_viz['Average_Syllables_per_Word'] = 0
KFP_viz_norm['Word_Count'] = 0
for i in range(len(KFP_viz)):
    #With cleaning
    KFP_viz['Average_Char_per_Word'][i] = textstat.avg_character_per_word(KFP_viz['TextForVisuals'][i])
    KFP_viz['Average_Syllables_per_Word'][i] = textstat.avg_syllables_per_word(KFP_viz['TextForVisuals'][i])
        

    #With cleaning + outlier removal
    KFP_viz_norm['Reading_Ease'][i] = textstat.flesch_reading_ease(KFP_viz_norm['TextForVisuals'][i])
    KFP_viz_norm['Word_Count'][i] = textstat.lexicon_count(KFP_viz_norm['TextForVisuals'][i])    
    KFP_df['Average_Sen_Length'][i] = textstat.avg_sentence_length(KFP_df['discourse_text'][i])
    KFP_viz_norm['Reading_Time'][i] = textstat.reading_time(KFP_viz_norm['discourse_text'][i])


# Note that the 'Average Sentence Length' uses the original text in its original form, as the sentence length would not be computable if we'd stripped punctuation;
# I am not using the processed text for 'Reading Time', but I am using the trimmed dataset; 
# I am using processed and trimmed sets for the other values. 
# I think that reading time and sentence length values will be more informative if we include all of the information about punctuation and stop-word usage. 

# In[18]:


import plotly.express as px
fig=px.histogram(data_frame=KFP_viz_norm,
                 x=KFP_viz_norm.Reading_Time,
                 marginal="violin",
                 color=KFP_viz_norm.discourse_type)

#Reading time is a statistic from textstat that estimates how long reading the text would take, assuming ~15ms per character.
fig.update_layout(title="Reading Time Variation with Respect to All Discourse Types:",
                  titlefont={'size': 25},
                  template='plotly_white'     
                  )
fig.show()


# So, here we can, for example, see that evidence tends to take the most time to read (on average), while there are quite a lot of claims in comparison to, say, positions.

# In[19]:


import plotly.express as px
fig=px.histogram(data_frame=KFP_viz_norm,
                 x=KFP_viz_norm.Reading_Ease,
                 marginal="violin",
                 color=KFP_viz_norm.discourse_type)

#See link below-- a metric of textual difficulty (negative values indicate extremely hard to make sense of)
fig.update_layout(title="Text Difficulty Variation with Respect to All Discourse Types:",
                  titlefont={'size': 25},
                  template='plotly_white'     
                  )
fig.show()


# Here, the outliers are in the negative direction (at least of what's left). The link below describes both the calculation this uses as well as the interpretation of the various scores. Roughly, the higher the score, the easier the text is to read, with a score of 90 being roughly commensurate with a 5th-grade reading level. My suspicion (given that the Harvard Law Review scores in the 30s on this scale) is that those descending from the peak of the curve are increasingly using bad punctuation, misspelled words, run-on sentences, and general incoherence, making them pretty inscrutable. [Read more.](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease)

# In[20]:


fig=px.histogram(data_frame=KFP_viz,
                 x=KFP_viz.Average_Char_per_Word,
                 marginal="violin",
                 color=KFP_viz.discourse_type)

fig.update_layout(title="Character Count Variation with Respect to Discourse Types:",
                  titlefont={'size': 25},template='plotly_white'     
                  )
fig.show()


# In[21]:


temp = KFP_viz.groupby('discourse_type').count()['discourse_id'].reset_index().sort_values(by='discourse_id',ascending=False)
fig = go.Figure(go.Funnelarea(
    text =temp.discourse_type,
    values = temp.discourse_id,
    title = {"position": "top center", "text": "Funnel-Chart of discourse_type Distribution"}
    
    ))

fig.update_layout(font = {'size' : 20})
fig.show()


# In[22]:


fig = px.bar(x=np.unique(KFP_viz["discourse_type"]), 
             y=[list(KFP_viz["discourse_type"]).count(i) for i in np.unique(KFP_viz["discourse_type"])], 
             color=np.unique(KFP_viz["discourse_type"]), 
             color_continuous_scale="Mint")
fig.update_xaxes(title="Classes")
fig.update_yaxes(title="Number of Rows")
fig.update_layout(showlegend=True, 
                  title={
                      'text':'Discourse Type Distribution', 
                      'y':0.95, 
                      'x':0.5, 
                      'xanchor':'center', 
                      'yanchor':'top'}, template="seaborn")
fig.show()


# In[23]:


fig=px.histogram(data_frame=KFP_df,
                 x= KFP_df['Average_Sen_Length'],
                 marginal="violin",
                 color=KFP_df.discourse_type)

fig.update_layout(title="Average Sentence Length for All Discourse Types:",
                  titlefont={'size': 25},template='plotly_white'     
                  )
fig.show()


# So, here, despite keeping punctuation and all intact, there are apparently a few texts that have virtually no periods, exclamations, or question marks, and just run on for 200-600 characters, distorting the dataset. 

# In[24]:


Trimmed_ASL = KFP_df.loc[(KFP_df['Average_Sen_Length'] <200) & (KFP_df['discourse_type'])]
fig=px.histogram(data_frame=Trimmed_ASL,
                 x= Trimmed_ASL['Average_Sen_Length'],
                 marginal="violin",
                 color=Trimmed_ASL.discourse_type)

fig.update_layout(title="Average Sentence Length for All Discourse Types (Trimmed):",
                  titlefont={'size': 25},template='plotly_white'     
                  )
fig.show()


# In[25]:


fig=px.histogram(data_frame=KFP_viz,
                 x=KFP_viz.Average_Syllables_per_Word,
                 marginal="violin",
                 color=KFP_viz.discourse_type)

fig.update_layout(title="Average Number of Syllables per Word with Respect to Discourse Type:",
                  titlefont={'size': 25},template='plotly_white'     
                  )
fig.show()


# In[26]:


fig=px.histogram(data_frame=KFP_viz_norm,
                 x=KFP_viz_norm.Word_Count,
                 marginal="violin",
                 color=KFP_viz_norm.discourse_type)

fig.update_layout(title="Word Count Distribution By Discourse Type:",
                  titlefont={'size': 25},template='plotly_white'     
                  )
fig.show()


# So, for the most part, our dataset appears to be complete, properly formatted, and ready for analysis. We aren't given much-- the text, and what type it belongs to. It is possible to reconstruct the entire students' essay (in some cases), and I believe that some of the winning models included those kinds of analyses in their models. However, I couldn't think of any helpful way to do so, so we won't look at those until the end. 

# ## Part 3: Introductory Methods in Natural Language Processing 
# 
# Resolving Encoding
# 
# Since the data is in such good shape, there's not a whole lot that we have to do in terms of fixing missing values, imputation, reformatting, and so on. A few items, however, will need to be addressed. 
# 
# Firstly, prior to this competition, I did not realize how many different styling indices existed for rendering text: utf, ACSII, unicode, and so on. While this didn't prove to be a huge problem, there were some oddball texts I discovered, and the following functions resolve all of that to ensure a smooth execution later on. (Tip o' the hat to Mr. DSML on Kaggle for providing the template: [Link](https://www.kaggle.com/code/brandonhu0215/feedback-deberta-large-lb0-619).)

# In[27]:


from typing import Dict, Tuple, List
import codecs
from text_unidecode import unidecode
def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]: return error.object[error.start : error.end].encode("utf-8"), error.end
def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]: return error.object[error.start : error.end].decode("cp1252"), error.end
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)
def resolve_encodings_and_normalize(text: str) -> str:
    text = (text.encode("raw_unicode_escape").decode("utf-8", errors = "replace_decoding_with_cp1252").encode("cp1252", errors = "replace_encoding_with_utf8").decode("utf-8", errors = "replace_decoding_with_cp1252"))
    text = unidecode(text)
    return text
    
KFP_df.discourse_text = KFP_df.discourse_text.apply(lambda x: resolve_encodings_and_normalize(x))


# Before we do much else, I want to go ahead and make a copy of the text in its original state, so that I can have it available for the feature engineering and other analyses prior to model-building, but after cleaning and processing the text. 

# In[28]:


KFP_df['discourse_copy'] = KFP_df.discourse_text


# In the process of completing this project, I encountered a great-many novel terms and concepts, both from the fields of linguistics and cognitive science, as well as from computer science, programming, and data analytics. Likewise, tricks-of-the-trade that would likely have never occurred to me to attempt were abundant, and I'll review and demonstrate some of those here. 
# 
# Firstly, let's import some of the more popular NLP-specific modules: 
# - NLTK: Natural Language Toolkit
# - Gensim
# - TextBlob & Spacy (I don't use them here, but I could have -- their functionality and implementation are very similar, so check it out if you're interested)
# 
# And I assume anyone reading this is already familiar with pickle, regex, and string. 

# In[29]:


import nltk
import string
import pickle as pkl
import gensim
import regex as re


# One of the first thoughts I had when I started working on this was about how to deal with contractions. If you're not careful, for one thing, when you .strip and .split and so on in Pandas, you can inadvertently create some oddball text-items. But furthermore, I don't see out-of-hand why contractions couldn't have some statistical implications. One module that I found I and liked is called 'contractions', and can be used in the following way: 

# In[30]:


import contractions
Demo = "Don't or Shouldn't or Can't or Won't?"
print(f" Original: {Demo}")
print('\n',f'Original (Split): {Demo.split()}')
Demo2 = strip_punctuation(Demo)
print('\n',f"Gensim Version: {Demo2}")
print('\n', f"Gensim (Split_PrePunctuation): {Demo}")
print('\n',f"Gensim (Split_PostPunctuation): {Demo2.split()}")
Demo3 = contractions.fix(Demo)
print('\n', f"Contractions Module: {Demo3}")
print('\n', f"Contractions Module (Split): {Demo3.split()}")


# In other words, using the contractions module allows you to completely delete punctuation, while not deleting any words, not retaining misspelled words, and not splitting the words along the apostrophe. I doubt it makes too much difference at the final analysis stage, but I like to have as much control as possible over what I keep and what I delete.
# 
# The next step is to strip out all of the punctuation marks, oddball characters, and capital letters.

# In[31]:


import contractions
contractions.add("i'm", 'i am')
KFP_df.discourse_text = KFP_df.discourse_text.apply(contractions.fix)

from gensim.parsing.preprocessing import strip_punctuation
KFP_df.discourse_text = KFP_df.discourse_text.apply(strip_punctuation)

KFP_df.discourse_text = KFP_df.discourse_text.apply(lambda x: x.lower())
KFP_df.discourse_text = KFP_df.discourse_text.apply(lambda x: x.strip())

from gensim.parsing.preprocessing import strip_multiple_whitespaces
KFP_df.discourse_text = KFP_df.discourse_text.apply(strip_multiple_whitespaces)


# In[32]:


#Output
KFP_df.discourse_text[1]


# Now, since any text model that we will ultimately use for classification will a vectorized version of any features we pass, it is necessary to remove what are known as **stopwords**: words that are ubiquitous, words that add little-to-no value or potential for distinction, and so on. These words will create a lot of noise and chaos in the final vector space, ultimately pointing our learner off-target. 

# In[33]:


from gensim.parsing.preprocessing import remove_stopwords
print(gensim.parsing.preprocessing.STOPWORDS)


# Some of these words, as you can see, are pretty unsurprising: a lot of prepositions ('of', 'in'), a lot of prefixes/affixes ('un', 're'), a lot of pronouns ('she', 'him'), a lot of psuedo-words ('ltd', 'co', 'ie'), and a lot of words that just appear in everything ('are', 'did', 'just', 'well', 'but', and so forth). 
# 
# On the other hand, some of them are odd to me ('latterly'?), Shakespearean ('hereupon', 'whence'), numeric ('fifty', 'eight'), or otherwise just...not what I expected ('mill', 'computer'). In any event, I presume the authors of the list have spent enough time researching and curating this list to have made a pretty good case for their inclusion. You can read more about the development of the list (at least according to Gensim) in 'Stone, Denis, Kwantes (2010).'

# In[34]:


KFP_df['discourse_text'] = KFP_df['discourse_text'].apply(remove_stopwords)


# The next step is to 'tokenize' the text. (Overly) Simply put, this means parsing the sentence into discrete lexemes (words) or graphemes (symbols, notation) so as to make the text amenable to later forms of processing or analysis. The RegexpTokenizer, courtesy of NLTK, uses regular expressions as the boundary at which it cuts the string. Some models, it should be mentioned, are optimized to work with a model-specific tokenizer, so select accordingly. 

# In[35]:


tokenizer = RegexpTokenizer(r'\w+')
KFP_df['discourse_tokens'] = KFP_df.discourse_text.apply(tokenizer.tokenize)
KFP_df['discourse_tokens'][22]


# Frequency Analysis is the process of analyzing documents for the frequency with which they contain some object of interest. In cryptography, this is used to decipher characters or words based on the probability of that word appearing in a given target text's language, for example. Here, we'll explore briefly the popularity of words in the text, and what -- if any -- implications they carry for this analysis. 

# In[36]:


from nltk.probability import FreqDist
#join tokens into strings                                                         #Note: only tokens greater than 2 char.
KFP_df['frequency_strings'] = KFP_df['discourse_tokens'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
#join ALL strings into a single string
discourse_words = " ".join([word for word in KFP_df['frequency_strings']])
#tokenize this monstrosity
document_tokens = tokenizer.tokenize(discourse_words)
#Make a counter dict from this composite string
fdist = FreqDist(document_tokens)


# In[37]:


fdist.most_common(10)


# In[38]:


top25 = fdist.most_common(25)
series_top25 = pd.Series(dict(top25))
fig = px.bar(y = series_top25.index,
            x = series_top25.values,)

fig.update_layout(barmode = 'stack',
                 yaxis = {'categoryorder':'total ascending'})

fig.show()


# We can see clearly that most of the words^ are either a) topical, specific to the essay prompt (e.g., 'states', 'votes), b) pretty-much-stop-words ('like'), or c) so general and ubiquitous they'll likely do more harm than good in signal-to-noise terms ('know', 'better'). We can deal with this by adding some additional qualifications to our token filter. We can also eliminate some oddball words that appear only once in the entire dataset, words that appear constantly in the datset, and anything that's not a word but a number.
# 
# 
# 
# (^If you want to bump-up your credibility in a crowd of non-NLP people, say 'Unigrams,' not 'words.' An 'N-gram' is the formal way of referring to a subsection of text that has length N: for example a 'trigram' is a subset of three words-- excuse me, unigrams-- like 'who even cares' or 'highfalutin academic talk' and so on.) 

# In[39]:


topical_unigrams = ['mona', 'lisa', 'electoral', 'college', 'landform', 'project', 'projects', 'venus', 'electors', 'phones',
                   'learning', 'votes', 'aliens', 'congress', 'constitutional', 'president', 'vote', 'classes', 'online',
                   'summer', 'election', 'students', 'student', 'school', 'extracurricular', 'life', 'mars', 'disctrict', 
                    'columbia', 'kerry', 'ran', 'national']

more_stopwords = ['like', 'the', 'i', 'want']

topical_unigrams+= more_stopwords

KFP_df.discourse_tokens = KFP_df.discourse_tokens.apply(lambda x: \
                                                        [token for token in x if not token.isnumeric() \
                                                         and fdist[token] >1 \
                                                         and fdist[token] < 25000 \
                                                         and token not in topical_unigrams]) 


# Finally, in the still-further interest of clearing out noise and reducing dimensionality, a common practice is to either 'Lemmatize', 'Stem', or 'Both' (kidding) the tokens. Both of these processes involve some melange of simplifying/affix-removing/de-conjungating/etc. the words to an infinitive form, a 'root', or something like that. As you can likely tell, I'm not 100%-clear myself on what the rules are, but some illustrative examples can clarify. (Tip o' the hat to the folks over at Turing.com for this helpful guide: [Link](https://www.turing.com/kb/stemming-vs-lemmatization-in-python).)

# In[40]:


from nltk.stem import PorterStemmer
PS = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer
SS = SnowballStemmer(language= 'english')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
illustrative_words = ['plays', 'playing', 'pharmacy', 'pharmacies', 'programmatic', 'badly']

print(f"Original: {illustrative_words}")
print('\n'f"Porter Stemmer-ed: {[PS.stem(i) for i in illustrative_words]}")
print('\n'f"Snowball Stemmer-ed: {[SS.stem(i) for i in illustrative_words]}")
print('\n'f"Word-Net-Lemmatizer-ated: {[lemmatizer.lemmatize(i) for i in illustrative_words]}")


# Since the computer largely interprets -- at least in some models -- the word-vectors (more later) for 'pharmacy' and 'pharmacies' as equivalent, the stemmer reduces noise and redundancy by abbreviating the word to some kind of least-common-denominator such that whatever it is paying attention to is identical for both. As I mentioned above, though, the exact process needed will depend, in part, on the model you choose and what you're trying to do. In this case, I'm going to stick with the lemmatized version since it seems to retain the most meaning, and since the stemmed versions print out so badli. 

# In[41]:


lemmatizer = WordNetLemmatizer()
for token_list in KFP_df.discourse_tokens:
    for token in token_list: 
        lemmatizer.lemmatize(token)


# In[42]:


KFP_df.discourse_tokens.head(5)


# A word cloud is an absolutely worthless graphic in terms of analytics, but the wordcloud module makes it super easy to do, and they look kind of cool, so what the heck? 

# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

lemmat_strings = KFP_df.discourse_tokens.apply(lambda x: ' '.join([i for i in x]))
lemmatized_tokens = ' '.join([i for i in lemmat_strings])

wordcloud = WordCloud(width = 600,
                     height = 400,
                     random_state = 2,
                     max_font_size = 100).generate(lemmatized_tokens)

plt.figure(figsize= (10, 7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off');


# ## Part 4: Hypothesis-Generation And Feature Engineering 
# 
# With all of the text parsing and preprocessing out of the way, we have a prepared set of data that's amenable to analysis. However, we don't have all that much non-textual content to work with. The statistics given by the textstat module above may be useful, but I have some suspicions that I'd like to explore at least, to see if we can squeeze anything further out of this. First off, I want to rid my df of the irrelevant features we created above.

# #### Punctuation Mark Analysis: 
# 
# While the rubric doesn't state that punctuation is considered in grading, that doesn't mean that it *isn't*. As discussed, all human graders are biased, and most are lovers of language and literature (at least essay-graders). It's hard to resist the tug of emotional-aesthetic revulsion when faced with horrid grammar. Likewise, the things the rubric *does* specify (complexity, valid argumentation, supportive evidence, clear logic, and so on) are, I'd wager, positively correlated with factors of intelligence, literacy, breadth-of-exposure to good writing, and attentiveness to school and detail. Thus the effective use of punctuation may not be *why* an excerpt scored highly, but it may still contribute to the recognition of such items. 
# 
# 1) Text-style (lol 'textile') writing-- as in 'text *messaging*'-- is frequently CRAZY!!!!! LIKE THIS!!!!!!!!!!! There's a reasonable number of exclamation points to use in a text of x-length, and that number may well be zero. But it would be odd if every sentence ended with one! Tell me I'm wrong! See how strange this is?! Thus, I'd guess their excessive use is an indication of weak writing. 

# In[44]:


KFP_df['exclamation_count'] = KFP_df.discourse_copy.apply(lambda x: len([i for i in x if i == '!']))


# 2) As Abe Lincoln never said, "The thing about quotation marks is they come in pairs." It seems to me that an odd number of quotation marks is a bad sign... Furthermore, the rubric specifies that effective excerpts reference external material and grab attention with witticisms and sagacicities. If one is using quotation marks (in pairs, at least), that probably means they are, well, quoting someone. Since this is specifically identified as valuable, we'll check for it. 

# In[45]:


KFP_df['quotation_marks'] = KFP_df.discourse_copy.apply(lambda x: len([i for i in x if i == '"']))


# 3) In general, at least one or two of the following marks should be present in pretty much any effectively written sentence. That said, their presence doesn't indicate much, but their absence does. Furthermore, these should increase in frequency more-or-less linearly with text-length. You simply can't, and shouldn't, avoid using apostrophes to annotate contractions, commas to separate clauses, and periods to terminate sentences. Even terse sentences require periods. A low-ratio of marks like these to words used means-- almost certainly-- run-on sentences, incoherent babble, and plain-ol' bad writing that you dont won't to be guilty of doin.

# In[46]:


basic_punctuation_marks = ['?',
                           ',',
                           "'",
                          '.']

KFP_df['basic_punctuation'] = KFP_df.discourse_copy.apply(lambda x: \
                                                                     len([i for i in x if i in basic_punctuation_marks]))


# 4) These punctuation marks are more subtle, and they tend to be underused. There are complicated rules dictating the use of colons: sometimes they are used for abrupt direction-changes, other times for implied conclusions, and yet other times for making lists. Parentheses are also tricky (and underused). (They're especially finnicky when used in standalone sentences.) Semi-colons and hypens aren't easy either; many writers mis-use them, or neglect them entirely. Likewise, the rubric suggests making reference to statistics in support of evidence, as well as in an attention-grabbing lead, but I'd reckon no more than 25% of students did. 

# In[47]:


effective_punctuation_marks = ['-',
                               '(',
                               ')',
                                '%',
                               ':',
                               ';'
                              ]
                               

KFP_df['positive_punctuation'] = KFP_df.discourse_copy.apply(lambda x: \
                                                                     len([i for i in x if i in effective_punctuation_marks]))


# 5) In much the same way as above, I think it's reasonable to think that the mere *presence* of a lot of punctuation is a good thing.  

# In[48]:


punctuation_marks = string.punctuation

def punctuation_diversity(text):
    score = 0
    marks = set(text)
    punctuation_markz = list(punctuation_marks)
    
    for mk in punctuation_markz: 
        if mk in marks:
            score+= 1 
    
    return score
KFP_df['punctuation_diversity_text'] = KFP_df['discourse_copy'].apply(lambda x: punctuation_diversity(x))


# #### Lexical Analysis:
# 
# I think a safe and simple premise, at least for exploratory analysis, is that some degree of probability that the essay/element has performed well or poorly can be revealed by the words used by the author. The following are some collections of words that I propose carry relevance to that end. 
# 
# 1) Complex Unigrams:I mentioned above that the competition guidelines put me in mind of *Integrative Complexity.* To my mind, certain words and ideas suggest this more distinctively than others. For example, words like 'whereas,' 'however,' and 'perspective' suggest that perhaps the author is considering counterarguments or taking other views into account. Words like 'therefore,' 'furthermore,' and 'prove,' suggest an argument is being built. 'Sources,' 'quote,' and 'research', 'percentage,' and so on suggest reference is being made to credible, external material. 

# In[49]:


complex_unigrams= ['procrastinate', 'whereas', 'detrimental', 'grasp', 'pursue', 'additionally', 'irrelevant', 'critical',
          'motivation', 'thus', 'however', 'assignments', 'assignment', 'source', 'sources', 'interests', 'asking', 'given',
          'skills', 'education', 'educational', 'ability', 'schooling', 'beneficial', 'allowing', 'instance', 'example',
            'therefore', 'excel', 'concepts', 'perspective', 'contradictory', 'allows', 'solutions', 'direction','forgoing',
            'view', 'viewpoint', 'confirmatory', 'confirm', 'evidence', 'proof', 'disprove', 'prove', 'contradict',
            'challenge', 'challenging', 'contradict', 'mislead', 'demonstrate', 'misleading', 'compromise', 'resolve']


# In[50]:


def complexity_eff(x):
    score = 0
    for item in x:
        if item in complex_unigrams:
            score +=1
    return score

KFP_df['count_E_tokens']= KFP_df['discourse_tokens'].apply(lambda x: complexity_eff(x))


# 2) Tricky Spelling: A quick Google search -- along with self-reflective composition -- shows that not all words are created equally in terms of easiness-to-spell. Since they are so frequently gotten wrong, I doubt that a human reviewer would penalize their inaccuracy (assuming they even noticed). However, if they are spelled *correctly* that suggests the author was either a) quite lucky or b) sufficiently verbally agile to get the spelling correct. In the latter case, correct spellings may be a sign of superior literacy. 

# In[51]:


tricky_spells_correct = ['absence', 'address', 'access', 'believe', 'beginning', 'privilege', 'separate', 'license',
                        'necessary', 'height', 'foreign', 'essential', 'receive', 'receiving', 'focused', 'though',
                        'through', 'unique', 'experience', 'experiences', 'occur', 'success', 'field', 'views', 'achieve']


# In[52]:


def spelling_eff(x):
    score = 0
    for item in x:
        if item in tricky_spells_correct:
            score +=1
    return score


KFP_df['effective_spelling']= KFP_df['discourse_tokens'].apply(lambda x: spelling_eff(x))


# 3) Not-So-Tricky Spelling: In contrast to (some) of the words above, some words are either a) not difficult to spell, or b) find misspellings that defy the imagination. I-before-E-trips-up-even-me (on occasion), but win sumwun spels perticlurlee bad, like a Chik-Fil-A cow, it's harder to excuse. 

# In[53]:


misspells = ['absense', 'adress', 'alot', 'beleive', 'cieling', 'calendur', 'begining', 'experiance', 'embarass', 'sience', 
            'seperate', 'wierd', 'truely', 'independant', 'goverment', 'hieght', 'foriegn', 'greatful', 'enviroment', 
             'privelege' 'libary', 'lisense', 'misterious', 'neccessary', 'peice', 'nieghbor', 'peolpe', 'electorals', 'stuf', 
             'alot', 'stuff', 'think', 'whats', 'pharaoh', 'activitys','ther', 'gonna', 'beacause', 
             'actaully', 'somone', 'selves', 'driveing', 'paragragh', 'moc', 'aswell']


# In[54]:


def sloppy_spelling(x):
    score = 0
    for item in x.split():
        if item in misspells:
            score += 1
    return score

KFP_df['count_misspelled_tokens']= KFP_df['discourse_copy'].apply(lambda x: sloppy_spelling(x))


# 4) Contraction Reaction: I dont think yall need something thats this obvious to be explained further, aint that right? Cant really help yourself...

# In[55]:


Bad_Contractions = ['cant', 'wont', 'isnt', 'aint', 'dont', 'werent', 'wernt', 'doesnt', 'thats', 'arent', 'couldnt',
                   'wouldnt', 'didnt', 'hadnt', 'Im', 'im', 'shouldnt', 'shes', 'hes', 'lets', 'id', 'hed', 'havent', 'ill',
                   'ive', 'Ive', 'Ill', 'theres', 'theyd', 'theyre', 'theyll', 'weve', 'wed', 'youve', 'youre', 'youd',
                    'whos', 'whove', 'wheres', 'whats']

KFP_df['contraction_errors_text'] = KFP_df['discourse_copy'].apply(lambda x: \
                                                                   len([i for i in x.split() if i in Bad_Contractions]))


# 5) Conservative Unigrams: This is a bit of a controversial addition, but I figured it was worth looking at. Given this is a university-sponsored competition, given many of the essays appeared to discuss political topics, and given increasing political division in America, I don't think it's unreasonable to suspect that politically conservative words might be associated with a strong reaction (negative or positive). 

# In[56]:


conservative_unigrams = ['conservative', 'bible', 'moral', 'christian', 'god', 'liberty', 'trump']

def conservatism(x):
    score = 0
    for item in x: 
        if item in conservative_unigrams:
            score+=1 
    return score

KFP_df['conservatism_score'] = KFP_df['discourse_tokens'].apply(lambda x: conservatism(x))


# 6) Overused Unigrams: It seems to me that an abundant presence of vague, hackneyed, or filler-type words might be a sign of poor rhetorical skill, unclear thinking, a lack of something substantial to say, or all of the above. 

# In[57]:


basic_unigrams = ['anybody', 'guys', 'basically', 'conclusion', 'like', 'the', 'want', 'i', 'very', 'whatever']

def complexity_inef(x):
    score = 0
    for item in x:
        if item in basic_unigrams:
            score += 1
    return score

KFP_df['basic_unigrams']= KFP_df['discourse_tokens'].apply(lambda x: complexity_inef(x))


# 7) Frequency Analysis: The frequency dictionary generated above didn't show any distinctive difference between the vocabularies of 'Adequate' essays and the vocabularies of 'Ineffective' ones. However, the 'Effective' essays had several hundred words that didn't appear in either of the other ratings. Some are no doubt idiosyncratic, topical, or otherwise not applicable to the test-set texts. However, the following list is some of which I suspect are more generalizable. 

# In[58]:


E_Additions = ['shown', 'systems', 'essential', 'effects', 'struggling', 'unable', 'bias', 'biased', 'notes',
              'pursue', 'sources', 'perspective', 'perspectives', 'previous', 'interaction', 'concept',
              'occur', 'success', 'solve', 'field', 'cases', 'available', 'general', 'quality', 'directly', 
              'strong', 'additionally', 'flaws', 'flaw', 'flawed', 'aspect', 'include', 'influence', 
               'relationships', 'considering', 'crucial', 'efficient', 'resources', 'resource', 'provide',
              'provides', 'furthermore', 'exploration', 'results', 'fully', 'ultimately', 'potentially']


# In[59]:


def E_xclusive(x):
    score = 0
    for item in x:
        if item in E_Additions:
            score += 1
    return score

KFP_df['E_Exclusive_tokens']= KFP_df['discourse_tokens'].apply(lambda x: E_xclusive(x))


# 8) Bigrams: As I started to think about bigram analysis, I quickly realized how out-of-hand this could get, so I've opted to only point to a handful that I think are particularly strong. (Plus this makes my computer run slowly to process, and that drives me bonkers.) 

# In[60]:


bigram_list = ['according to',
          'for example',
           'for instance',
           'may claim',
           'despite that',
           'fair point',
           'valid point',
           'valid objection',
           'research shows',
               'light of',
               'you might',
               'they think']

tuple_list = []
for i in bigram_list:
    b = tuple(i.split())
    tuple_list.append(b)


# In[61]:


KFP_df['bigrams'] = KFP_df['discourse_copy'].apply(lambda x: list(nltk.bigrams(x.split())))
KFP_df['bigrams_score'] = KFP_df['bigrams'].apply(lambda x: len([i for i in x if i in tuple_list]))
KFP_df.drop(columns = ['bigrams'], axis =1, inplace=True)


# #### Style Analysis
# Given that this project evaluates the effectiveness of argumentation, I thought there may be some stylistic features that are more-or-less unique to/characteristic of weak arguments.
# 1) ALL CAPS!!!!! (AND EXCLAMATION POINTS!!!!!!!!!SEE ABOVE!!!!!!!) Rather than articulating anger and detailing frustration, all-caps writing suggests the written equivalent of a shouting match. 

# In[62]:


def caps(x):
    score = 0
    for i in x.split(): 
        if i.isupper():
            score += 1
            
    return score
KFP_df['caps_text'] = KFP_df['discourse_copy'].apply(lambda x: caps(x))


# 2) Relatedly, the use of the ad-hominem -- personal insults-- rather than subtle analysis of flaws in reasoning is a classic fallacy that would, on presumes, be met with aversion by graders. 

# In[63]:


ad_hominemlst = ['idiot', 'stupid', 'dumb', 'loser', 'losers', 'idiots', 'morons', 'moron', 'liar', 'liars']

def ad_hominem(x):
    score = 0
    for item in x:
        if item in ad_hominemlst: 
            score+=1
    return score

KFP_df['ad_hominem'] = KFP_df['discourse_tokens'].apply(lambda x: ad_hominem(x))


# 3) The appearance of a highly infrequent word seems to me to be a bad sign -- at least in this context. There are positive situations in which an extremely uncommon word would be a good thing: a spelling bee, a highly technical journal or article, or in one of Christopher Hitchens' essays. In this context, though, I suspect a highly infrequent word is one that is either a) misspelled, b) non-English, c) completly misused, or d) um...not a word (e.g., an onomatopoeia, a slang term, etc.). (Note, we have to use the original discourse_text variable, rather than the tokens, since we have already removed all tokens that appear less than once.)

# In[64]:


KFP_df['weird_tokens']= KFP_df['discourse_copy'].apply(lambda x: \
                                                               len([item for item in x.split() if fdist[item] == 1]))


# 4) I'm not sure that it's exactly 'incorrect' to have a text replete with numeric tokens, but I find it distracting and a bit juvenile. When I see 'there are 44 seats in the house' instead of 'there are forty-four seats in the house', it doesn't feel quite right to me. I'm not sure Strunk & White would concur, but I still think i'ts worth exploring. 

# In[65]:


def numerical(text):

    score = 0
    for word in text.split():
        if word.isdigit():
            score +=1
    return score

KFP_df['count_numerical_text'] = KFP_df['discourse_copy'].apply(lambda x: numerical(x))


# 5) Sentiment Analysis is another common NLP task, and is blessedly simple to implement using NLTK's API. Psychological research has shown a bias amongst the literati towards work that is negative in tone -- not so much writing that is angry or hostile, but writing that is pessimistic, that covers or describes injustice, disaster, war, political discord, and so on. The idea is that such work is more 'serious' than other content, and I can see the same thing happening in politically charged high-school essays.  

# In[66]:


from nltk.sentiment import SentimentIntensityAnalyzer
disc_sent_analyzer = SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')
KFP_df['polarity'] = KFP_df['discourse_text'].apply(lambda x: disc_sent_analyzer.polarity_scores(x))


# #### Standardization
# For almost all of the features added above, the *presence* of the features and the *count* of the features may not be particulalry illuminating. What I mean is that given the considerable variance in discourse *length* any of the given figures as a count could be quite misleading. For example, if a text was a paragraph-long, the presence of some relatively uninformative words wouldn't be a bad sign, and if it was a sentence-long, we wouldn't expect much diversity of punctuation. On the other hand, if it was a paragraph-long and had *no* punctuation (or very little), that would be a big red flag. That in mind, I think the ratio of these features to either the total number of characters or the total number of tokens in the text is a much more informative metric.  
# Before we take those measurements, however, I need to make sure to avoid requesting division by zero...

# In[67]:


def counting(x):
    alpha = len([i for i in x])
    if alpha == 0:
        alpha+=0.1
    
    return alpha

KFP_df['Characters_In_Text'] = KFP_df['discourse_copy'].apply(lambda x: counting(x))
KFP_df['Words_In_Text'] = KFP_df['discourse_text'].apply(lambda x: counting(x))
KFP_df['Tokens_In_Text'] = KFP_df['discourse_tokens'].apply(lambda x: counting(x))


# Relatedly, Claude Shannon's great insight was in realizing that natural language is dense with redundancy. Much of our speech can be condensend into vastly smaller spaces by avoiding repetition ("Avoid repetition! Avoid repetition!" - E.B. White), by not using lots of letters when a few will do the job ("Eschew superfluous verbiage" - Mark Twain), and by eliminating *predictable* components of speech and language. For example, psyclgsts hve shwn tht we cn rd frly cmplx sntcs evn whn mst vwls are absnt. Cryptographers, likewise, make short work of unsurprising passwords, codes, and on so on simply by counting the number of repeated characters in a masked or encrypted document. (Hint: in English, it's probably 'e'.) That in mind, having clipped all of the stop-words, dull and topical terms, white-space, and so forth, and whittled our way down to tokens that aren't ubiquitous in the document-space, the ratio of what's left to what we started with seems like it may be indicative of how much useful, original, and informative content was embedded in the text to begin with. 

# In[68]:


#How much is left: How much we started with:
KFP_df['informative_proportion'] = KFP_df.Tokens_In_Text/KFP_df['Characters_In_Text']

#How much of what's left that's *good* : How much we started with - punctuation, etc.:
KFP_df['proportion_E_tokens'] = KFP_df['count_E_tokens']/KFP_df['Words_In_Text'] 

#How much of what's left is not-so-good: How much we started with - punctuation, etc.:
KFP_df['proportion_I_tokens'] = KFP_df['basic_unigrams']/KFP_df['Words_In_Text']


# Likewise, we really need comparable metrics for some of the other features, like exclamation marks and so on: 

# In[69]:


#Odd Number of quotation marks
KFP_df['Odd_Number_Quotes'] = KFP_df['quotation_marks'].apply(lambda x: 1 if x%2 != 0 else 0)

#Surfeit of exclamation points
KFP_df['exclamation_proportion'] = KFP_df.exclamation_count/KFP_df['Characters_In_Text']

#Summary of poor writing features:
bad_signs = ['count_misspelled_tokens', 
             'contraction_errors_text',
             'caps_text',
             'basic_unigrams', 
             'ad_hominem', 
             'weird_tokens',
            'count_numerical_text',
            'exclamation_count',
            'Odd_Number_Quotes']

KFP_df['bad_signs_score'] = KFP_df[[i for i in bad_signs]].sum(axis = 1)


# ## Part 5: KDD/EDA/Mining/FeatureSelection/OtherSmartSoundingDataWordsAndAcronyms
# 
# Let's begin by taking a look at exactly what all it is that we have, then sorting it by data-type. We can then explore removing redundant or colinear features, collapsing similar items, binning or otherwise discretizing some of the features if needed, and comparing distributions by discourse effectiveness score. 

# In[70]:


#We don't need the following variables for analysis: 
to_drop = ['TFV_length', 'essay_id','TextForVisuals', 'frequency_strings']
KFP_df.drop(columns = [i for i in to_drop], axis = 1, inplace = True)

#We need the various 'count'-related items to all be int, not float. 
transform = ['Words_In_Text', 'Tokens_In_Text']
KFP_df[[i for i in transform]] = KFP_df[[i for i in transform]].astype('int64')

#Grouping features of common data types:
TextFeats = ['discourse_text', 'discourse_tokens', 'discourse_copy']
IntFeats = KFP_df.select_dtypes(include= ['int64'])

#The proportion features need to remain floats: 
FloatFeats = ['informative_proportion', 
              'proportion_E_tokens', 
              'proportion_I_tokens', 
              'exclamation_proportion', 
              'Average_Sen_Length']

#Discourse type will work best as a categorical feature
KFP_df.discourse_type = KFP_df.discourse_type.astype('category').cat.codes

#Other: the polarity_score, which comes to us in the form of a dictionary of values: 
otherFeats = ['polarity']


# In[71]:


from yellowbrick.features import Rank2D

visualizer1 = Rank2D(algorithm = 'pearson', 
                     size = (996,696),
                    title = "Correlation Between Continuous IVs")

visualizer1.fit_transform(KFP_df[[i for i in IntFeats]])
visualizer1.show();


# Unsurprisingly, the 'tokens', 'words,' and 'characters'-in-text have a high level of correlation, and there's considerable correlation with the punctuation_diversity feature as well. 
# We can also see a high positive correlation between the number of n-grams that were numeric, and the 'bad_signs' score. 

# In[72]:


Redundant = ['Words_In_Text', 'Tokens_In_Text', 'Characters_In_Text']
#For these, we can collapse them into a single metric
KFP_df['Length'] = KFP_df[[i for i in Redundant]].sum(axis = 1)
KFP_df['Length'] = KFP_df['Length'].apply(lambda x: round(x/3), 2)
KFP_df.drop(columns = [i for i in Redundant], axis = 1, inplace = True)
#Since this in -- and otherwise strongly correlated with -- the 'bad_signs', we can get rid of it. 
KFP_df.drop(columns = ['count_numerical_text'], axis = 1, inplace = True)
#We can just drop this as well.
KFP_df.drop(columns = ['basic_punctuation'], axis = 1, inplace = True)
#The count of quotation marks was only collected to find texts with an odd number thereof
KFP_df.drop(columns = ['quotation_marks'], axis = 1, inplace = True)


# In[73]:


from yellowbrick.features import Rank2D

visualizer1 = Rank2D(algorithm = 'pearson', 
                     size = (596,296),
                    title = "Correlation Between Continuous IVs")

visualizer1.fit_transform(KFP_df[[i for i in FloatFeats]])
visualizer1.show();


# Nothing seems to be problematic amongst those variables. 

# In[74]:


KFP_df.shape


# 29 features is still an awful lot. Since we've collapsed the 'bad_signs' -- each of which in isolation may not be a particularly strong predictor-- I think it's probably okay to drop those. 
# We also still have the 3 variations on the text, even though only the tokenized and lemmatize version will likely be needed during model-building. 

# In[75]:


to_drop_2 = ['discourse_text', 'discourse_copy']
bad_signs.remove('count_numerical_text')
to_drop_2+= bad_signs
KFP_df.drop(columns = [i for i in to_drop_2], axis = 1, inplace = True)


# Yellowbrick has another very cool feature-selection visual regressor that we can use to make an estimate as to the predictive potential of our features.

# In[76]:


x = KFP_df[[i for i in FloatFeats]]
y = KFP_df.discourse_effectiveness.to_list()
from yellowbrick.model_selection import FeatureImportances
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class = 'auto', solver = 'liblinear')
visualizer3 = FeatureImportances(model, 
                                 stack = True, 
                                 relative = False, 
                                 xlabel = '1')
visualizer3.fit(x,y)
visualizer3.show();


# Clearly the informative-proportion and proportion of tokens that are in the list associated with effectiveness are distinctly more predictive. 

# In[77]:


KFP_df.drop(columns = ['proportion_I_tokens', 'Average_Sen_Length', 'exclamation_proportion'], axis = 1, inplace = True)
Non_numeric_Feats = ['discourse_id', 'discourse_type', 'discourse_effectiveness', 'discourse_tokens', 'polarity']
FloatFeats2 = ['informative_proportion', 'proportion_E_tokens']
RemainingIntFeats = [i for i in KFP_df.columns if i not in Non_numeric_Feats and i not in FloatFeats2]


# In[78]:


x = KFP_df[[i for i in RemainingIntFeats]]
y = KFP_df.discourse_effectiveness.to_list()
from yellowbrick.model_selection import FeatureImportances
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class = 'auto', solver = 'liblinear')
visualizer3 = FeatureImportances(model, 
                                 stack = True, 
                                 relative = False, 
                                 xlabel = '1')
visualizer3.fit(x,y)
visualizer3.show();


# Well, that's interesting. I didn't anticipate that the 'conservatism' score would be nearly as important as it seems here, nor that it would predict significantly across effectiveness scores. I also find it a little surprising that length -- whether using the composite character-token-word metric, or using textstat's average sentence length-- has so little impact. We'll go ahead and drop it, since the word-vectors will more-or-less reflect length anyway. 
# 
# There doesn't seem to be a need for the punctuation metrics much at all, but we can collapse them. The count_E and E_exclusive appear to be capturing the same things in the same directions, so we can just take the mean of those. 

# In[79]:


to_drop_4 = ['Length']
#Collapsing Effective Tokens
Effective_Tokens = ['count_E_tokens', 'E_Exclusive_tokens']
KFP_df['Effective_Tokens'] = KFP_df[[i for i in Effective_Tokens]].sum(axis = 1)
KFP_df['Effective_Tokens'] = KFP_df[['Effective_Tokens']].apply(lambda x: round((x/2), 2))
#Punctuation
Punctuation = ['punctuation_diversity_text', 'positive_punctuation' ]
KFP_df['Punctuation'] = KFP_df[[i for i in Punctuation]].sum(axis = 1)
KFP_df['Punctuation'] = KFP_df['Punctuation'].apply(lambda x: round(x/2), 2)
#Dropping unnecessary variables
to_drop_4 += Effective_Tokens + Punctuation
KFP_df.drop(columns = [t for t in to_drop_4], axis = 1, inplace = True)


# We haven't looked yet at the polarity scores, which are a separate data structure (dictionary), but they are formatted as shown in the cell below. I think it will be easier to grasp what-- if any -- value they add if we split out the dictionary into it's types. 

# In[80]:


KFP_df.polarity[1]


# In[81]:


KFP_df['Neg'] = KFP_df['polarity'].apply(lambda score_dict: score_dict['neg'])
KFP_df['Neu'] = KFP_df['polarity'].apply(lambda score_dict: score_dict['neu'])
KFP_df['Pos'] = KFP_df['polarity'].apply(lambda score_dict: score_dict['pos'])
KFP_df['Compound'] = KFP_df['polarity'].apply(lambda score_dict: score_dict['compound'])
KFP_df.drop(columns = ['polarity'], axis = 1 ,inplace = True)


# In[82]:


import plotly.express as px
df = KFP_df
fig = px.box(df, 
             x = 'Compound', 
             color = 'discourse_effectiveness', 
             notched= True,  
             title = 'Quantiles of Mean Polarity by Discourse Effectiveness',
            labels = {'Compound': "Compound Polarity Score"})
fig.update_traces(quartilemethod = 'exclusive')
fig.update_layout(title = {'xanchor': 'left'})
fig.show()


# Apart from a very slight inclination towards greater positivity in effective essays, there doesn't appear to be much in the way of distinctiveness nor consistency in this feature. 

# In[83]:


KFP_df.drop(columns = ['Neg', 'Neu', 'Pos', 'Compound'], axis = 1, inplace = True)


# In[84]:


FinalFeatures = KFP_df.select_dtypes(include= ['int64', 'float64'])


# In[85]:


from yellowbrick.features import Rank2D

visualizer1 = Rank2D(algorithm = 'pearson', 
                     size = (596,296),
                    title = "Correlation Between Continuous IVs")

visualizer1.fit_transform(KFP_df[[i for i in FinalFeatures]])

visualizer1.show();


# Apart from an obvious degree of correlation between Effective_Tokens and Proportion_E_tokens, everything seems pretty independent. While all of the engineered features are integers or floats, the majority aren't well-conceived as continuous predictors, with the exception of 'informative_proportion': 

# In[86]:


KFP_df['informative_proportion'].hist(figsize = (8,6), xrot = 45, bins = 45)
plt.show()


# In[87]:


#Stratified by Effectiveness Score
fig=px.histogram(data_frame=KFP_df,
                 x=KFP_df.informative_proportion,
                 marginal="violin",
                 color=KFP_df.discourse_effectiveness)

fig.update_layout(title="Distribution of Informative Proportion By Discourse_Effectiveness:",
                  titlefont={'size': 25},template='plotly_white'     
                  )
fig.show()


# The remaining variables, it seems to me, are better binned -- many of them are zero, and given the variance in text-length, it seems like the mere *presence* of them may be the primary signal. 

# In[88]:


KFP_df.proportion_E_tokens = KFP_df.proportion_E_tokens.apply(lambda x: 1 if x > 0 else 0)


# In[89]:


discBadSigns = KFP_df['proportion_E_tokens']
discEff = KFP_df['discourse_effectiveness']
newb = pd.concat([discBadSigns, discEff], axis = 1)
newbCT6 = pd.crosstab(discBadSigns, discEff, normalize = 'index')
newbCT6
plot5 = newbCT6.plot.barh(stacked = True, color = ['gold', 'forestgreen','firebrick'])
plot5.set(xlabel= 'Effectiveness Proportion', ylabel = 'Use of Effective Tokens')
plot5.legend(bbox_to_anchor = (1.05, .6));


# In[90]:


def count_cat(x): 
    cat = 0
    if x in (1,2):
        cat+=1
    if x in (3,4): 
        cat+=2
    if x in (5,6):
        cat+=3
    if x in (7,8): 
        cat += 4
    if x > 9:
        cat+= 5
    return cat

KFP_df.Effective_Tokens = KFP_df.Effective_Tokens.apply(lambda x: count_cat(x))
KFP_df.effective_spelling = KFP_df.effective_spelling.apply(lambda x: count_cat(x))
KFP_df.bad_signs_score = KFP_df.bad_signs_score.apply(lambda x: count_cat(x))
KFP_df.Punctuation = KFP_df.Punctuation.apply(lambda x: count_cat(x))
KFP_df.conservatism_score = KFP_df.conservatism_score.apply(lambda x: count_cat(x))
KFP_df.bigrams_score = KFP_df.bigrams_score.apply(lambda x: count_cat(x))


# In[91]:


binnable = ['effective_spelling', 
            'conservatism_score', 
            'bigrams_score', 
            'bad_signs_score', 
            'Effective_Tokens', 
            'Punctuation']
KFP_df[[i for i in binnable]].describe()


# In[92]:


KFP_df.describe()


# In[93]:


fig, ax = plt.subplots(3,2,figsize = (20,30))
for variable, subplot in zip(binnable, ax.flatten()):
    sns.countplot(x = KFP_df['discourse_effectiveness'], ax = subplot, hue= KFP_df[variable])
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# So, taking these 1-by-1: 
# 1) Effective spelling is linearly distributed -- in that it increases from ineffective-to-adequate-to-effective-- but it is so sparse, it's unlikely to be very helpful. The big challenge for the algorithm will be in using the presence or absence to differentiate at the edges of Ineffective-Adequate, and Adequate-Effective. Telling ineffective from effective will likely be much easier, but if there's not a distinctive separation between adequate and one of the extremes, we can't get anything out the feature. I'm going to drop this due to not being sufficiently dense. 
# 2) Bigrams is virtually indistinguishable when present at all. Drop.
# 3) Effective tokens has the same problem as [1]. Drop.
# 4) Conservatism may be informative, but it's probably too sparse to be useful. 
# 5) Bad signs looks like it's worth exploring in more depth. We are really more interested in proportions than counts, so we can investigate that with a stacked bar chart. 
# 6) Punctuation, as well, is hard to say.

# In[94]:


KFP_df.drop(columns = ['effective_spelling', 
                       'bigrams_score', 
                       'conservatism_score', 
                       'Effective_Tokens'], 
            axis = 1, 
            inplace = True)


# In[95]:


discBadSigns = KFP_df['bad_signs_score']
discEff = KFP_df['discourse_effectiveness']
newb = pd.concat([discBadSigns, discEff], axis = 1)
newbCT3 = pd.crosstab(discBadSigns, discEff, normalize = 'index')
newbCT3
plot3 = newbCT3.plot.barh(stacked = True, color = ['gold', 'forestgreen','firebrick'])
plot3.set(xlabel= 'Effectiveness Proportion', ylabel = 'Bad Signs Score')
plot3.legend(bbox_to_anchor = (1.05, .6));


# So, bad signs pretty much gets progressively associated with ineffectiveness. That's a good thing. But where the proportions change, it's coming from adequate-- Effective essays are proportional irrespective of how many bad signs they have. 

# In[96]:


discBadSigns = KFP_df['Punctuation']
discEff = KFP_df['discourse_effectiveness']
newb = pd.concat([discBadSigns, discEff], axis = 1)
newbCT4 = pd.crosstab(discBadSigns, discEff, normalize = 'index')
newbCT4
plot4 = newbCT4.plot.barh(stacked = True, color = ['gold', 'forestgreen','firebrick'])
plot4.set(xlabel= 'Effectiveness Proportion', ylabel = 'Punctuation Score')
plot4.legend(bbox_to_anchor = (1.05, .6));


# So, a punctuation score of 2, 3, or 4 is the sweet-spot -- Effectives tend to concentrate there. Too much = ineffective; too little = adequate. I think this is also worth keeping. 
# 
# That leaves us with 1 continuous predictor, naturally scaled to 0-1, and roughly normal in distribution: informative_proportion.
# One given categorical variable: discourse_type.
# One text variable: discourse_tokens.
# One binary variable: 'proportion_E_tokens'.
# One positively-oriented ordinal variable: Punctuation.
# One negatively-oriented ordinal variable: bad_signs_score.
# An identifier: discourse_id.
# And one dependent variable: discourse_effectiveness.

# ## Part 6: Model-Building and Evaluation
# 
# I won't make this textbook beta-test any longer by including the full code for every model, but I'll give you enough here to get a sense of what I did. All of the model specs are available on my github if you'd like to explore them in greater detail. Better yet, go check out my Tip o' the Hat links, since in many cases mine were adaptations of related configurations, and those were made by much more skilled and competent analysts. 
# 
# A basic challenge at this point in a project of this kind is discerning not only what variables to keep and what to drop, but also figuring how to combine so many different data types into a single model. As mentioned above, we have very nearly every different data type you can have, not all of which are well-suited to the same kind of algorithm or model. 
# 
# 1) Sklearn Simple Models
# 
# - Logistic Regression (Text-Only)
# - Multinomial Naive Bayes (Text-Only)
# - Support Vector Machine (Text-Only)
# - Random Forest Classifier (Eng. Features- Only)
# - AdaBoost Classifier (Eng. Features-Only)
# - XGBoost Classifier (Eng. Features-Only) 
# 
# 2) Sklearn Integrated Features Pipelines
# 
# 3) Sklearn Enesembles
# 
# 4) Bigram Analysis 
# 
# 5) Features-as-Text Models
# 
# 6) Doc2Vec

# So, in the early stages, I want to see what the unadorned text does in unadorned models, to get a sense of both a) which models work more or less effectively for the content we have, and b) to establish a baseline for comparing more complex models. If you look on Kaggle, you'll see, for example, ~200 submissions that have a log loss > .80, which is what we achieve with just an out-of-the-box logistic regression below. The competition had two tracks-- one for accuracy and one for efficiency, so establishing this baseline with an essentially instant model is helpful in considering the accuracy/efficiency tradeoff. 

# In[97]:


#Train-Test-Split
from sklearn.model_selection import train_test_split as tts
#Processing and combining
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
#Exploratory Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#Evaluative programs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss


# In[98]:


#Join tokens into complete strings to feed to vectorizer
KFP_df['model_tokens'] = KFP_df.discourse_tokens.apply(lambda x: ' '.join([i for i in x]))
#Train-Test Split
x = KFP_df.model_tokens
y = KFP_df.discourse_effectiveness.to_list()
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.3, random_state = 42)


# Model 1: Logistic Regression:
# The CountVectorizer assembles word vectors with a frequency-based dictionary which builds a count-matrix.  
# The Tfidf_Transformer performs a logarithmic transform to normalize the count-matrix. The 'tf' indicates 'term-frequency' and the 'idf' indicates the 'inverse document frequency.' Simply put, it prevents extremely common words from drowning out infrequent words in terms of feature weight. 
# The Pipeline is a chaining method, allowing multiple sequential processes to be simply listed, rather than expressly performed item-by-item. Note that while fit_transform would typically be applied to the training data, whereas only 'fit' would be applied to the test data, this sequence is not necessary when using the pipeline. 

# In[99]:


text_clf1 = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression(max_iter = 5000)),])

text_clf1 = text_clf1.fit(x_train, y_train)

pred = text_clf1.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
labels = ['Effective', 'Ineffectve', 'Adequate']
pred = text_clf1.predict_proba(x_test)
evaluate = log_loss(y_test , pred) 
print(f"Log-Loss = {round(evaluate, 3)}")


# The log-loss metric printed at the bottom is the metric used in the competition evaluation: it permits classification performance in imbalanced datasets-- like this one. We have, for example, ~57% 'Adequate' scores, so the heuristic 'Always predict Adequate' would perform better than 50/50 (or 33/33/33), but would not add anything. With log-loss, lower is better, indicating the probabilities assigned to each target class for each observation are significantly favoring the correct class. .808 corresponds, as we can see, to a ~8-10% accuracy improvement over the naive heuristic; the winning submissions had a log-loss of ~.554.
# 
# The next baseline model is a Multinomial Naive Bayes, run through the same pipeline.

# In[100]:


text_clf2 = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])

text_clf2 = text_clf2.fit(x_train, y_train)

pred = text_clf2.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
labels = ['Effective', 'Ineffectve', 'Adequate']
pred = text_clf2.predict_proba(x_test)
evaluate = log_loss(y_test , pred) 
print(f"Log-Loss = {round(evaluate, 3)}")


# Here, again, we can sense the imperfect alignment of log-loss and accuracy: while we only decreased in accuracy by one percent, our log-loss jumped ~5 percentage points. The MNB must have generated less confident probabilities, despite ultimately concluding on approximately the same number of categorical guesses. 
# 
# Our last baseline attempt is a support vector machine, which does reasonably well (though not as well as the logistic regressor), but which doesn't have a built-in method for outputting a probability estimate and computing the log-loss. 

# In[101]:


text_clf3 = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),])

text_clf3 = text_clf3.fit(x_train, y_train)

pred = text_clf3.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
labels = ['Effective', 'Ineffectve', 'Adequate']


# Next, we can try iteratively adding in some of the features we engineered, to see if they add anything. Decision Tree models can serve as a good non-parametric approach for using unscaled and disparate data types. Note that we are converting the target -- discourse effectiveness -- from it's given form ("Adequate, Effective, Ineffective") to type('category') and then taking cat.codes. This is essentially a categorical encoding, just the extra-easy way. 

# In[102]:


featsX = ['discourse_type', 
            'informative_proportion', 
            'proportion_E_tokens', 
            'bad_signs_score', 
            'Punctuation']

x = KFP_df[[i for i in featsX]]
y = KFP_df.discourse_effectiveness.astype('category').cat.codes
x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.3, random_state = 42)


# In[103]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

rfc1 = rfc.fit(x_train, y_train)

pred = rfc1.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
labels = ['Effective', 'Ineffectve', 'Adequate']
pred = rfc1.predict_proba(x_test)
evaluate = log_loss(y_test , pred) 
print(f"Log-Loss = {round(evaluate, 3)}")


# In[104]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
ada1.fit(x_train, y_train)

print('Training Accuracy: {:.2f}'.format(ada1.score(x_train, y_train)))
print('TEST Accuracy:  {:.2f}'.format(ada1.score(x_test, y_test)))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = ada1.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
pred = ada1.predict_proba(x_test)
evaluate = log_loss(y_test , pred) 
print(f"Log-Loss = {round(evaluate, 3)}")


# In[105]:


from xgboost import XGBClassifier

from sklearn.metrics import f1_score, accuracy_score
xgb3 = XGBClassifier(random_state = 42, num_class = 3)
xgb3.fit(x_train, y_train)
pred = xgb3.predict(x_test)
cf = confusion_matrix(y_test, pred)
print(cf)
print(classification_report(y_test, pred))
pred = xgb3.predict_proba(x_test)
evaluate = log_loss(y_test , pred) 
print(f"Log-Loss = {round(evaluate, 3)}")


# In[106]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.heatmap(cf/np.sum(cf), annot = True, fmt= '.2%', cmap = 'Greens')

ax.set_title('Confusion Matrix for FeatureUnion XGB\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('\nActual Values\n')

ax.xaxis.set_ticklabels(['Adequate', 'Effective', 'Ineffective'])
ax.yaxis.set_ticklabels(['Adequate', 'Effective', 'Ineffective'])
plt.show()


# I find it interesting that the xgboost -- using only the features we gave it-- is more capable of identifying the effective items than the ineffective: it only gets 1/16 correct there. 
# 
# We can see what it found useful using the following:

# In[107]:


from xgboost import plot_importance
from matplotlib import pyplot

plot_importance(xgb3)
pyplot.show();


# In[108]:


feat_gains = xgb3.get_booster().get_score(importance_type = 'gain')
pyplot.bar(feat_gains.keys(), feat_gains.values());
pyplot.xticks(rotation = 90);


# Curiously, the 'gains' (the graph on the bottom) suggests that the model finds the proportion of effective tokens to be its most valuable predictor. Additionally, we perform almost as well as the text models (not that those are especially stellar, but still) using only the extracted features. There's some debate as to which of the XGBoost importance-types ('weight', 'gain', 'total_gain', 'cover', and 'total_cover') are most valuable and informative. The coverage metric deals with how many observations were influenced by a particular split, while the gains deals with the relative strength of the feature as a predictor. These are exactly inverted with respect to the 'proportion_E_tokens': it has the least coverage (probably because there are so many zero-values), but it's presence or absence makes a big difference. This reinforces the observation I made above about this competition hinging upon the 'edge-cases': items near the cutoff thresholds from Adequate-to-Effective/Ineffective. It appears that the proportion of strong tokens does indeed help to differentiate Effectives from Adequates. But the bad signs and the informative proportion (though widely *applicable*) don't add much in terms of distinguishing Ineffectives from Adequates. 
# 
# The 'SHAP' (Shapley Additive Explanations) model for determining feature importance was an entirely separate approach that was strongly endorsed on many forums and in many articles. There is a module for computing this at the link below, but I am totally unfamiliar with the (game-theoretic) approach, and have added learning it to my to-do list, but haven't implemented it here. (Tip o' the Hat to the folks at [Link](https://shap.readthedocs.io/en/latest/index.html).)
# 
# Anyway, a different approach to model-comparison is given below, wherein bigram frequency is also considered, and then model-selection is assisted by a cross-validation process that iteratively tries different sklearn algorithms. I include it because it is cool and because it illustrates some valuable tools/approaches, but I didn't end up using it. Since the process involves re-mapping some features we have already converted, as well as the vectorizer replicating processes we've already applied, I'm switching back to the original dataset for this example. (Tips o' the Hat to [Link1.](https://medium.com/tokopedia-data/step-by-step-text-classification-fa439608e79e) and [Link2.](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f))

# In[109]:


dfcopy = pd.read_csv('train.csv')


# In[110]:


dfcopy['effectiveness_id'] = dfcopy['discourse_effectiveness'].factorize()[0]
from io import StringIO
category_id_df = dfcopy[['discourse_effectiveness', 'effectiveness_id']].drop_duplicates().sort_values('effectiveness_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['effectiveness_id', 'discourse_effectiveness']].values)


# In[111]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

vec_tfidf = TfidfVectorizer(ngram_range = (1,2), 
                            sublinear_tf = True,
                            analyzer = 'word', 
                            norm = 'l2',
                            max_df = 15000,
                            min_df = 15,                  #Min_df/max_df == the min/max times an ngram has to occur
                            encoding = 'latin-1',
                            stop_words = stop_words)

discourse = vec_tfidf.fit_transform(dfcopy.discourse_text).toarray()


# In[112]:


labels = dfcopy.effectiveness_id
discourse.shape


# In[113]:


from sklearn.feature_selection import chi2
import numpy as np

N = 5

for discourse_effectiveness, effectiveness_id in sorted(category_to_id.items()):
    discourse_chi2 = chi2(discourse, labels == effectiveness_id)
    indices = np.argsort(discourse_chi2[0])
    feature_names = np.array(vec_tfidf.get_feature_names_out())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]


# In[114]:


print("'{}':".format(discourse_effectiveness))
print("  Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
print("  Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))


# In[115]:


X_train, X_test, y_train, y_test = tts(dfcopy['discourse_text'], 
                                       dfcopy['discourse_effectiveness'], 
                                       random_state = 0)


# In[116]:


models = [
    RandomForestClassifier(n_estimators = 200, max_depth = 3, random_state = 0),
    LinearSVC(),
    MultinomialNB(), 
    LogisticRegression(random_state = 0, max_iter = 1000)
]

CV = 5
cv_df = pd.DataFrame(index = range(CV * len(models)))


# In[117]:


from sklearn.model_selection import cross_val_score

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, discourse, labels, scoring = 'accuracy', cv = CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
        
cv_df = pd.DataFrame(entries, columns= ['model_name', 'fold_idx', 'accuracy'])


# In[137]:


import seaborn as sns

sns.boxplot(x = 'model_name', 
            y = 'accuracy', 
            data = cv_df)

sns.stripplot(x = 'model_name',
              y = 'accuracy',
              data = cv_df,
              size = 15, 
              jitter = True,
              edgecolor = 'black',
              linewidth = 2)

plt.show();

cv_df.groupby('model_name').accuracy.mean()


# In[138]:


from sklearn.model_selection import train_test_split as tts
model = LogisticRegression(random_state = 42, max_iter = 1000)

X_train, X_test, y_train, y_test, indices_train, indices_test = tts(discourse,
                                                                   labels,
                                                                   dfcopy.index,
                                                                   test_size = 0.33,
                                                                    random_state = 0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[139]:


from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize = (8,6))
sns.heatmap(conf_mat,
           annot = True,
           fmt= 'd',
           xticklabels = category_id_df.discourse_effectiveness.values,
           yticklabels = category_id_df.discourse_effectiveness.values)

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show();


# In[140]:


from IPython.display import display

for predicted in category_id_df.effectiveness_id:
    for actual in category_id_df.effectiveness_id:
        if predicted != actual and conf_mat[actual, predicted] >= 10:
            print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
            display(dfcopy.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['discourse_effectiveness', 'discourse_text']])
            print('')

model.fit(discourse, labels)


# In[141]:


from sklearn.feature_selection import chi2

N = 10
for discourse_effectiveness, effectiveness_id in sorted(category_to_id.items()):
    indices = np.argsort(model.coef_[effectiveness_id])
    feature_names = np.array(vec_tfidf.get_feature_names_out())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
    
print("# '{}':".format(discourse_effectiveness))
print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names = dfcopy['discourse_effectiveness'].unique()))


# So, this process gave us the strongest prediction so far, but not by much. A downside to all of these iterative model-tests is that it takes quite a long time to generate all of the scores -- and adding the bigrams in for analysis makes it even worse. I'll point out, once again, that this approach manages to identify >80% of Adequates, nearly 50% of Effectives, but only 1/6 Ineffectives. 

# 
# #### Sklearn FeatureUnion Pipelines
# 
# Having found baseline performances, the next logical step seems to be to try integrating those features and feeding them into a single predictor (or predictive ensemble). The first approach to doing so is using a pipeline that looks at *both* text and the engineered features, using appropriate transformations for each, then unionizing them to make all of the columns amenable to simultaneous analysis. 

# In[142]:


#Modules
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer


# In[143]:


#Classes to pull text, numeric features
class TextTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key
        
    def fit(self, X, y = None, *parg, **kwarg):
        return self
    
    def transform(self, X):
        return X[self.key]
    
class NonTextTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X[[self.key]]


# In[144]:


#Central Pipeline
fp = Pipeline([
    ('features', FeatureUnion([
        ('model_tokens', Pipeline([
            ('transformer', TextTransformer(key = 'model_tokens')),
            ('vectorizer', TfidfVectorizer(ngram_range = (1,1), 
                                            sublinear_tf = True,
                                            analyzer = 'word', 
                                            norm = 'l2',
                                            max_df = 15000,
                                            min_df = 2,                              
                                            encoding = 'latin-1'))])
        ),
        ('informative_proportion', Pipeline([
            ('transformer', NonTextTransformer(key= 'informative_proportion'))])),
        ('proportion_E_tokens', Pipeline([
            ('transformer', NonTextTransformer(key= 'proportion_E_tokens'))])),
        ('bad_signs_score', Pipeline([
            ('transformer', NonTextTransformer(key= 'bad_signs_score'))])),
        ('Punctuation', Pipeline([
            ('transformer', NonTextTransformer(key= 'Punctuation'))])),
        ('discourse_type', Pipeline([
            ('transformer', NonTextTransformer(key= 'discourse_type'))
        ]))    
                                ])
    ),
#Here    
    ('xgb3', XGBClassifier(objective = 'multi:softmax', random_state = 42, num_class = 3))
         ])


# In[145]:


#GridSearch, k-fold
param_grid = {'clf__n_estimators': np.linspace(1, 100, 10, dtype=int),
              'clf__min_samples_split': [3, 10],
              'clf__min_samples_leaf': [3],
              'clf__max_features': [7],
              'clf__max_depth': [None],
              'clf__criterion': ['gini'],
              'clf__bootstrap': [False]}

kfold = StratifiedKFold(n_splits=7)
scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
refit = 'F1'

#XGBoostGridSearch
search_space = [
  {
    'xgb3__n_estimators': [50, 100, 150, 200],
    'xgb3__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'xgb3__max_depth': range(3, 10),
    'xgb3__colsample_bytree': [i/10.0 for i in range(1, 3)],
    'xgb3__gamma': [i/10.0 for i in range(3)],
  }
]

kfold = KFold(n_splits=2, random_state=42, shuffle = True)

scoring = {'AUC':'roc_auc', 'Accuracy': make_scorer(accuracy_score)}


# In[146]:


#train_test_split
x = KFP_df[['model_tokens', 
            'informative_proportion', 
            'discourse_type',
           'bad_signs_score',
           'Punctuation',
           'proportion_E_tokens']]

y = KFP_df['discourse_effectiveness'].astype('category').cat.codes
x_train, x_test, y_train, y_test = tts(x, y, random_state = 42, test_size = 0.3)


# Due to the runtime -- and corresponding difficulties in terms of configuring the Jupyter-Book, I'm commenting out the execution of the Grid Search algorithm below.
```{py}

grid = GridSearchCV(
                  fp,
                  param_grid=search_space,
                  cv=kfold,
                  scoring=scoring,
                  refit='AUC',
                  verbose=1,
                  n_jobs=-1
                    )



model = grid.fit(x_train, y_train)

gsv_best = model.best_estimator_
print(gsv_best)
gsv_pred = model.predict(x_test)
acc = accuracy_score(rf_pred, y_test)
print(acc)
labels = ['Adequate', 'Effective', 'Ineffective']
pred = model.predict_proba(x_test)
evaluate = log_loss(y_test ,pred) 
print(evaluate)

# Using the framework above, all that's required to test different models is to specify them from '#here' down. That is, replace the model with the desired one, and then fill the appropriate grid-search elements in. I've actually experimented with a variety of these, but just include one as an illustrative example. Likewise, I experimented with GridSearch, RandomSearch, and melange of other cross-fold validation tools, but I think this is sufficient to grasp the essence of the process. No need to rehash every single instance of failed experimentation. 
# 
# The second approach is making non-text features *into* text that is appended to the word-vectors we pass to the model. 

# In[148]:


def integrated_text_features(x):
        
    Integrated_Text = []
    Integrated_Labels = []

    for index, row in x.iterrows():

        combined1 = ''
        combined1+= 'Bad Signs Score: {:}, Informative Proportion: {:}, Part of Text: {:}. '.format(row['bad_signs_score'],
                                                                                      round(row['informative_proportion'],3),
                                                                                      row['discourse_type'])
        combined1+= ''.join([i for i in row['model_tokens']])
        combined1 = combined1.lower()
        Integrated_Text.append(combined1)
        Integrated_Labels.append(row['discourse_effectiveness'])
        
    IText = pd.Series(Integrated_Text)
    ILab = pd.Series(Integrated_Labels)
    
    IPrepped = pd.concat([IText, ILab], axis = 1, keys = ['Text', 'Label'])
        
    return IPrepped


IPrepped_df = integrated_text_features(KFP_df)
IPrepped_df['Text'][1]


# Along with the output above, I also experimented with using text that hadn't been processed, using descriptors instead of numbers, and so on: 

# In[149]:


KFP_df2 = pd.read_csv("train.csv")
KFP_df2 = KFP_df2.merge(KFP_df, how = 'inner', left_index = True, right_index = True)
KFP_df2 = KFP_df2[['discourse_text', 
                   'discourse_effectiveness_x', 
                   'discourse_type_x', 
                   'informative_proportion', 
                   'bad_signs_score' ]]
KFP_df2.rename({'discourse_effectiveness_x': 'discourse_effectiveness',
               'discourse_type_x' : 'discourse_type',
               'discourse_text' : 'model_tokens'}, axis =1, inplace = True)

effectiveness_map = {0: 'Excellent',
                     1: 'Very Good', 
                     2: 'Good', 
                     3: 'Acceptable',
                     4: 'Poor',
                     5: 'Very Poor'}
KFP_df2['bad_signs_score'] = KFP_df2['bad_signs_score'].map(effectiveness_map)

IPrepped2 = integrated_text_features(KFP_df2)
IPrepped2['Text'][0]


# This approach was inspired by a wonderful set of blog posts, so Tip o' the Hat to Chris McCormick, Ken Gu, and the team at Multi-Modal Toolkit [Link.](https://mccormickml.com/2021/06/29/combining-categorical-numerical-features-with-bert/)
# 
# This appealed to me in that it was concise, and it gave me the impression of 'giving the computer my opinion.' Unfortunately -- at least with respect to the models I was able to configure fully with Kaggle's notebooks (more later)-- when I compared my highest-performing neural network classifier's performance on the original text and the annotated text, just leaving all of this out yielded (slightly) better accuracy. 
# 
# Finally, I tried a model using the 'doc2vec' approach -- in essence it extends the word-vectorization process to include 'tags' which add a layer of similarity grouping. When we split training and testing data, normally, the model we select treats the *outcome* as a tag in this sense. The model analyses all of the features, looks at the labels we've given it, and tries to define a function that sends inputs to the appropriate label (the dependent variable). In the doc2vec process, this is essentially done *at the step of word-embedding.* The model decides how to configure the vector space itself with a similar training process, linking documents of common tags in such a way that the vector space reflects *that* similarity, not just the similarity of words. The example below hopefully makes this process clear. Learn more at -- and Tip o' the Hat to-- the following links: 
# 
# [Link1.](https://towardsdatascience.com/implementing-multi-class-text-classification-with-doc2vec-df7c3812824d)
# 
# [Link2.](https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4)
# 
# [Link3.](https://fzr72725.github.io/2018/01/14/genism-guide.html)
# 
# [Link4.](https://towardsdatascience.com/how-to-vectorize-text-in-dataframes-for-nlp-tasks-3-simple-techniques-82925a5600db)
# 
# 

# In[151]:


#I'm reimporting the original file here so as not to add any unwanted modifications to the in-progress dataframe. 
df = pd.read_csv('train.csv')

#Some dependencies
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
import multiprocessing

# All of the preprocessing used above, just condensed. 
df.discourse_text = df.discourse_text.apply(contractions.fix)
df.discourse_text = df.discourse_text.apply(strip_punctuation)
df.discourse_text = df.discourse_text.apply(lambda x: x.lower())
df.discourse_text = df.discourse_text.apply(lambda x: x.strip())
df.discourse_text = df.discourse_text.apply(strip_multiple_whitespaces)
df['discourse_text'] = df['discourse_text'].apply(remove_stopwords)


# In[152]:


#First, we split the dataset, keeping the *final* target (effectiveness) in the test set. 
from sklearn.model_selection import train_test_split as tts
train, test = tts(df, test_size = 0.3, random_state = 42)

import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


# In[153]:


#Here, we indicate that all text sequences should be tagged with effectiveness Scores, to configure similarity based on outcome
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['discourse_text']), tags= [r.discourse_effectiveness]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['discourse_text']), tags=[r.discourse_effectiveness]), axis=1)


# In[154]:


#Now, we create the document vectors, and the algorithm learns to build the space in consideration of scores
cores = multiprocessing.cpu_count()
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors


# In[155]:


# *Now* we split the training and testing instances for the classifier
y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)

logreg = LogisticRegression(max_iter = 5000, n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(f"Testing Accuracy: {accuracy_score(y_test, y_pred)}")
pred = logreg.predict_proba(X_test)
evaluate = log_loss(y_test , pred) 
print(f"Log-Loss = {round(evaluate, 3)}")


# This is an interesting idea, and there's research suggesting it can be implemented to impressive effect. However, it's not something I've explored in much detail, and I don't feel like I understand it particularly well. That being the case, I'm not at all convinced the model couldn't be coaxed into performing better if it was tuned and customized by someone with greater mastery of this technique. 

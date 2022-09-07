#!/usr/bin/env python
# coding: utf-8

# ## Appendix 2: Some Frustrations
# 
# Briefly, there were parts of this competition which rubbed me the wrong way. 
# 
# Kaggle is an excellent concept, a wonderful group of folks, and a justly renowned educational platform. But it's not perfect. I really wish they had somewhat better explanations/tutorials/etc. for navigating their tools from the perspective of a non-computer-science-major. It's not at all obvious, for example, how to proceed when 'Cuda is out of memory', when 'notebook threw exception' (but no further details), or when 'internet must be disabled' (despite Kaggle not having the module otherwise available) is the feedback you get. There were multiple times where I just had to give up on model variations because despite them running cleanly in Jupyter, they wouldn't run in Kaggle; or despite them running in Kaggle, they wouldn't run in the submission. Based on the discussion boards, mods' contributions, and site updates, they seem to be aware, at least, that some of these things are adversely impacting a lot of users, but so far none of the answers provided worked for me (or either I was doing them wrongly -- a very real possibility, but if I, a motivated and well-educated user, can't figure them out, doesn't that just reinforce my point?).
# 
# In any event, my biggest frustration was with the dataset itself. I'll explore that briefly here, and please let me know what you think!

# In[1]:


import pandas as pd
gt = pd.read_csv('train.csv')


# In[2]:


gt.head(5)


# So, here we see the blocks of text, the discourse type, and the corrsponding score assigned. Nothing appears off. Given this dataset is a subset of another Kaggle dataset from a previous competition, it's presumably been checked and double-checked by hundreds of people. But looking more closely we find: 

# In[3]:


gt['length'] = gt.discourse_text.apply(lambda x: len([i for i in x]))


# In[4]:


gt['words'] = gt.discourse_text.apply(lambda x: len([i for i in x.split()]))


# In[5]:


Characters = 'This sentence, including spaces and punctuation marks, is of length:'
char_count = len(Characters)
Alternate = 'The shorter example.'
cc2 = len(Alternate)
print(Characters, char_count)
print(f"'{Alternate}' (Excluding this annotation and the quotation marks, this sentence has length {cc2}) ")


# Okay, so a given sentence of mildly low complexity is ~70 characters, while a complete sentence of near-maximal brevity -- while still being of complete and grammatically correct structure--is around 20. 
# 
# My instinct is that, given the rubric's criteria, it would be hard to be considered adequate using fewer characters than my example sentence above. Well..

# In[6]:


gt[gt.length < 20]


# 265 example-texts have a character count of 20 or fewer characters! As we can see above, these are not -- in many cases-- even complete *phrases*, much less sentences remotely resembling the examples given. 
# Nevertheless, these were largely rated to be adequate! For example: 

# In[7]:


print(gt['discourse_effectiveness'][11], '\n')

print(f"The entirety of observation eleven's discourse Text: '{gt['discourse_text'][11]}'  ")


# The entirety of this item's discourse_text is the word 'stress.' (Period in original). Let's recall what the rubric has to say about a 'claim' and how it is assessed: 
# 
# >Adequate: 
# >- Description: The claim relates to the position but may simply repeat part of the position or state a claim without support.
#     The claim is moderately valid and acceptable. 
#    
# > - Example: Position: "Every individual owes it to themselves to think seriously about important matters, no matter the difficulty".
#     Claim: " It is important to think seriously about important matters although some people do not do this". 
#     
# Note that the claim is contextually defined: it should be interpreted in terms of its relationship to the preceding position statement. Below, we can use the essay ID t reconstruct the original essay in its entirety:

# In[8]:


import simple_colors
exampleEssay = gt.loc[gt['essay_id']== '00944C693682']
for index, row in exampleEssay.iterrows():
    if row.discourse_text.startswith('With') or row.discourse_text.startswith('stress'):
        row.discourse_text = simple_colors.yellow(row.discourse_text, 'bold')
    print(f'(Discourse Type: {row.discourse_type})','\n', row.discourse_text,'\n', f'(Score= {row.discourse_effectiveness})', '\n')


# So, here we can see that the text is complete in itself. The student wrote: 
# 'Why would they all agree, one might ask. Well, there are plenty of reasons. Stress.' 
# 
# Presumably, they intended something to the effect of a list or heading: 
# - 'Why do people agree? Many reasons: stress, exhaustion, ...
# 
# Or:
# 
# - 'There are many reasons people agree. Reason one: Stress.
# Or something like that. 
# 
# 'Well, there are many reasons' is the actual 'claim'; 'stress' is basically a heading. But the former got identified as a 'position,' the single word 'stress' as the claim, and the subsequent discussion of car-less communities as the evidence. 

# This -- in conjunction with the multifarious examples that follow, lead me to think that the supervisory inputs are contradictory and incompatible with the rubric used to guide this dataset's development. These kinds of errors mislead both the analyst and the learner. In order for any algorithm to meaningfully learn there must be genuine and consistent distinctions. When a supervised learner gets: 
# 
# - Height | Description
# - 5'1"   | Tall
# - 5'2"   | Tall
# - 5'1    | Short
# - 6'1    | Tall
# - ...
# 
# there's no 'there' there. 
# 
# And indeed for human learner it's much the same: a person learning English would infer that the word 'height' must mean *some other feature*, but certainly not how tall people are. The instances being called 'tall' have no relationship with the instances' distance from head-to-ground. *If* some tangential or spurious link can be discovered, that may be even worse: it would mean the outputs from the machine indicate sharing our meaning in some sense, while in fact they have merely latched onto some completely unrelated factor invisble without the synchronous computation of a million partial derivatives.

# In[9]:


gt[gt['length'] < 60]


# Over 12% of the entire corpus consists of sentences, phrases, fragments, and discrete words that are shorter than my example. Of these, as you can see below, the majority were assessed to have adequately met the description (800  were even 'Effective'!).
# 
# Now, given this is a subset of a former Kaggle competition's dataset, I guessed that perhaps what happened was something was explained in the previous conversation that I was not aware of. But that page is still live [https://www.kaggle.com/competitions/feedback-prize-2021/overview/description] and it suggests very plainly that we also are looking for complete sentences (at least) to comprise the discourse element.

# In[10]:


len(gt.loc[(gt['length']<60) & (gt['discourse_effectiveness']=='Adequate')])


# In[11]:


len(gt.loc[(gt['length']<60) & (gt['discourse_effectiveness']=='Effective')])


# In[12]:


gt.loc[(gt['length']<60) & (gt['discourse_effectiveness']=='Effective') & (gt['discourse_type']=='Evidence')]


# I don't see how 'Let them grow up.' can have effectively met the criteria described above.
# 
# In any case, these aren't just oddballs.
# 
# Things don't look great when we shift from characters to words, either.

# In[13]:


Words = 'This sentence, excluding punctuation marks, contains the following number of words:'
word_count = len([i for i in Characters.split()])
print(Words, word_count)


# In[14]:


gt[gt['words'] < 10]


# 10% of all discourse_texts are of fewer than 10 words, most are 'Adequate' and -- at least of those visible-- consist of fragments of decontextualized words. 

# In[15]:


#Example of an effective type Claim: 
gt.discourse_text[46]


# As shown below, none of these are without a corresponding rating, since no ratings are absent from the dataset.

# In[16]:


gt.discourse_effectiveness.value_counts()


# In[17]:


print(len(gt))
print(gt.discourse_effectiveness.value_counts().sum())


# Likewise, we can see below that there are duplicates of the discourse_text: 76 to be precise. Since that number is so small, perhaps these are just goofs.

# In[18]:


duplicates = gt[gt.discourse_text.duplicated(keep=False)].sort_values(by="discourse_text")
duplicates.head(10)


# But that doesn't seem to be the case. Take a look at the 3rd/4th entry: it is exactly the same text, but it has been classfied as two different discourse-types. Item 5/6 shows that 'Big states' is both 'Adequate' *and* 'Ineffective'! The last four elements show that 'I agree' is of equal sophistication as 'I agree with the principal', which (to me) is hard to understand.  
# 
# Now, we could, I suppose, just remove all of these and continue by dismissing these several thousand cases and focusing on the more intact texts. 

# In[19]:


#only effective texts with more than 60 characters.
longer_texts = gt.loc[(gt['length'] > 60) & (gt['discourse_effectiveness']=='Effective') & (gt['discourse_type']== 'Counterclaim')]

#Let's take a look at a couple of (non-random) examples to illustrate my point.
#for i in longer_texts.discourse_text[43]:
 #   print(i, '\n')
    
print(longer_texts.discourse_text.values[36], '\n')    
print(longer_texts.discourse_text.values[26])


# But...
# 
# According to the grading rubric, an Effective counterclaim is: 
# >...reasonable and relevant. It represents a valid objection to the position.
# 
# While an Adequate one is: 
# 
# >...not quite a reasonable opposing opinion, or it is not closely relevant to the position. 
# 
# And this gets to the second point: who decides what is 'reasonable,' what is 'relevant,' and what is 'valid'?
# 
# To me, these two excerpts are just apples and oranges. The first one is obviously shorter, less informative, less thoughtful, rhetorically weaker, and barely constitutes a counterclaim (or at least a counter-*argument*) at all -- it's closer in my mind to a 'Claim.' The latter argument builds a case, legitimizes the point of view, and discusses the legitimate and plausible consequences that worry opposing points-of-view. 
# 
# Consider, another excerpt, and contrast it with the first example above: 

# In[20]:


Adequate_Comparison = gt.loc[(gt['length'] > 60) & (gt['discourse_effectiveness']=='Adequate') & (gt['discourse_type']== 'Counterclaim')]

print(Adequate_Comparison.discourse_text.values[199])


# Again, it's hard for me to see why this text is less effective than the example above. It is longer, more descriptive, and more complex. It is a little clunky with the 'projects to do'-bit, but the above one is clunky as well ('can argue'). In terms of ambiguity, validity, relevance, informativeness, and so on, *I* would judge this example to be a better element than the first. 
# 
# Now compare *that* Adequate text with this one: 

# In[21]:


print(Adequate_Comparison.discourse_text.values[52])


# This is incorrectly punctuated, uses incorrect capitalization, has multiple misspellings, introduces irrelevance ('335 horses'), is informal ('isn't for you') and -- most importantly-- *isn't a counterclaim*. It's a position, or maybe a claim, but unless the preceding sentence was describing some melange of things a critic might point out, this appears to be an opinion held by the author. How is this comparable to the item above?
# 
# I won't beat this dead horse further. My point, though, is that there appears to be a good bit of the messiness and impulsivity that human graders are prone to -- for example, being impressed/in a good mood/hungry/liking the student/fearing the parent/not wanting the headache... (or the opposite). When you're trying to meet your friends at the bar in time for the game, and you've got a stack of 30,000 essays to label and grade, you just default to 'Adequate', then when you see something that is even mildly impressive (or the name of a student you're fond of, etc., etc.), you throw-in an occasional 'Effective.' When you're tired or bored or spiteful and/or you come across a particularly awful text, you throw it in the ineffective pile.
# 
# What that results in is really 5 strata: 
# 
# - 1) Truly bad essays
# - 2) Truly good essays
# - 3) Truly Average essays
# - 4) Truly Average essays marked good or bad willy-nilly
# - 5) Truly good or bad essays marked average by default.
# 
# And the effect of those last two groups is that the waters become muddied -- The machine can pick between 1 and 2, at least pretty well, but since it's trained to recognize as 'Average' examples which are truly effective or ineffective, it just defaults basically *everything* to adequate. Only the most extreme cases of goood/bad are labeled as such.
# 
# Notice the indices in the code: I've not gone beyond triple-digits. Recall that there are over 36,000 texts in this document...

# I suppose you could make the case that the ambiguity and inconsistency and so on are part-and-parcel to the nature of this type of data or this type of project, and that the goal isn't to make the model necessarily *good* at evaluating essays, only to make the model *comparable to* the average of judgments made by the graders. But as a former teacher, I can tell you that humans are quite prone to sloppiness, incoherence, arbitrariness, and inconsistency. Ignoring the 'no there there' problem (i.e., that there *is* no meaningful pattern here, and thus any model will be useless in terms of generalizability), imagine what it would look like to deploy such a model. Do we really want -- or consider it in any way to be progress -- to have a machine-- one that informs schools, publishing houses, standardized exam makers, and so on -- that 'correctly' marks both dangling clauses full of misspellings *and* reasonable (if stylistically modest) counterclaims as both being adequate compositions? 
# 
# Put differently, it seems to me that any model fit to this dataset would accomplish the exact opposite of what we'd want in such a product. That is, we'd presumably want to outsource to a computer work that humans would do sloppily, lazily, in a biased manner, in an inconsistent manner, and so forth. If my eyes have glazed over after essay 400 and my mind has turned to such a soup that I'm unable to distinguish or discriminate any better than what's shown in these excerpts, no computer should be told to try and capture exactly how I'm thinking in that context! When we've trained, in the past, models on text that is racist, sexist, or otherwise discriminatory, we've ended up with racist and sexist AI. If we train models on text that reflects human errors or inconsistencies or whatever, then we'll have a grader that's just as bad as the error-prone human it's meant to replace. Of course, a neural network *can* discover *any* function -- even a chaotic, disorderly one-- but that doesn't mean it's any *good*. Any deployable model needs to have consistent parameters for identifying good from bad in a generalizable way, and if it is fit to such an idiosyncratic, slapdash dataset, it will yield a correspondingly chaotic framework. Consistently outputting nonsense may be consistent, but it's still nonsense. 
# 
# It seems, further, that AI which is trained on anything other than the best of human judgement will be a merely faster and less accountable source of errors, not a solution to them. We source the substance of AI's 'thought' through human-created data. Only the very best of this data ought to be captured in the algorithms we generate, and only be deployed after we've carefully confirmed this to be the case. That's the only way to ensure consistency, accuracy, and unbiased evaluation. Having mined this dataset for the better part of two months, and having expectation after expectation disconfirmed in model-testing, I'm not sure I'm convinced this is that sort of dataset, and I'm skeptical that any model derived from it is useful in terms of deployment. I certainly wouldn't want to be graded by it. 
# 
# I think the intention of this project is very noble and worthwhile, but without better data, I doubt it will do much in terms of ameliorating the limitations of present writing-feedback tools. The performance of this task requires both normative and aesthetic judgments, and Thanksgiving dinner with your family is all that's required to prove that objectivity with respect to what constitutes 'soundness', 'validity', 'effectiveness', 'logical coherence', 'evidence', and so on is not definable. What's more, even when agreement can be achieved about what works conceptually, there is an almost infinite number of ways in which it may be shared. Hemingway and Shakespeare may both be literary geniuses, but there's hardly a single quantifiable thing in their work that indicates that shared capacity. It is not the instantiation on paper, but the affective state brought about in human minds that distinguishes -- and links-- their work. Effective natural language processing is as much about human minds -- intentions, feelings, inclinations, sarcasm, bias -- as it is about the actual words that are present. This requires sensitivity to not only the structural elements of analysis (linguistics, statistical methods) -- the "processing," if you will --  but also to the "natural"-- the way humans actually use it, the way one individual's thoughtful remark is another's inconsiderate snarkiness, or the way one person's common sense is another's blatant falsehood. When you pair this with the way that language is inherently ambiguous, contextually dependent, and inseparable from models of the world and one's experiences therein, it seems unlikely that the statistical sophistication will be sufficient on its own to achieve what this project sets out to.
# 
# I mentioned above that I thought web-scraping for essays to use would be a better approach to building what the competition seems to be looking for. I think this would ameliorate at least some of the dataset's issues. For one thing, when an essay is selected to be *featured*, it is because that dataset is likely indisputably an example of the score it seeks to model. That level of reliability -- graded once by the tired, overworked grader, then implicitly graded again by the author of the website's post-- would ensure that the fickleness and caprice of the human grader wasn't mixed in with the reliable examples. Additionally, the diversity of topic-prompts that would be represented in this way would ensure that topical noise didn't confuse the learner, forcing it to focus on structural elements of the essay. Lastly, a 'segment identifier' algorithm could be built alongside the scoring algorithm, allowing the learner to 'recognize' complete instances of essay portions, rather than having the dataset be peppered with single-word elements. 

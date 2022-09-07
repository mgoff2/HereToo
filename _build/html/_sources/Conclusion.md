### Conclusion

```python
import cv2 as cv
import glob

imdir = 'Kaggle.png'
ext = ['png']

files = []
[files.extend(glob.glob(imdir + '*.'+e)) for e in ext]

images = [cv2.imread(file) for file in files]
from IPython import display
display.Image('Kaggle.png')

                    #A Portrait of the Author as a Humbled Man.
```


![png](output_0_0.png)
    

### Conclusion

Well, that's pretty much it for the datasest and the model-design. As mentioned at the start, I placed around 1000/1600, so that's obviously suboptimal. On the other hand, this being my first attempt at NLP, first attempt at Kaggle, and first attempt at building any sort of meaningful neural network, I think the learning I achieved more than justifies what time I took in working through all of this. My confidence in Pandas/Text Processing/Numpy/Sklearn/Matplotlib+Seaborn+Plotly has shot way up, and I've discovered a wide-range of novel Python tools, related to all manner of topics and tasks, and useful for all sorts of projects beyond this one or even NLP. Additionally, while the competition is over, I still want to play around with this dataset, and with the purpose of the competition (i.e., NLP that can assess argumentative strength, as well as related concepts of coherence, validity, and so on). Some things that I didn't explore but want to include:

- Tools: AutoKeras | SpacySentenceBert | Sklearn preprocessing items KBinsDiscretizer, RobustScaler, and others. | Doc2Vec | Seq2Seq models 
- Methods: Rule-based and Human-in-the-Loop model designs | Genetic Algorithms | Techniques for resolving class imbalances
- Data science-specific 'best practices' for OOP and functional programming, as well as contributions to open source projects.
- Problem-Specific Interests: I describe, in the sections following this, some frustrations that I had with Kaggle/the dataset/the competition rules, etc., one of which has to do with the quality and structure of the data. It seems to me that scraping websites for 'example' essays (shouldn't be too hard, what with education and standardized testing blogs and so forth) that are scored in a comparable manner to those in this dataset. That would, I think, make the whole thing more robust and reliable.
  (With respect to the latter, I found myself doing a lot of self-repetition, which I knew was taboo in programming (and now understand why), but didn't trust my skill enough to design anything too complex.)

Along with completing this model-competition dimension of the project, I've also gone down the rabbit hole on more than a few occasions exploring NLP generally and the psycho-philosophical side of language and AI. That's the topic of the next section, and it's largely an essay, with minimal code/analytics, so proceed at your discretion.

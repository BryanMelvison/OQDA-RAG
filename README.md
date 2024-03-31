# OQDA-RAG
I am trying to implement an RAG from scratch, so this is the pipeline I will be following, we will follow the implementation of **DPR** (Dense Passage Retrieval) according to this paper: https://arxiv.org/pdf/2004.04906.pdf 

The whole RAG pipeline will consist of:
1. Training the two transformer-encoder models, one for encoding representations of the question title and body, another for encoding the answer corpus.
2. Once models are fine-tuned, POST the corpus embeddings and answer corpus to the MongoDB Database
3. Inference: Once everything has been posted, inference can be done by following the formats.
4. Results to LLM -> For this segment, we will be using the output from our DPR as an input to the LLM of our choice, then it will summarize our findings based on the answers retrieved.

So far we've done the first **three** steps. As inference time on CPU is quite slow(around 45 seconds), whereas with GPU, it will take around (14 seconds). Therefore, for future work, I will look into FAISS (Facebook AI similarity search), for further efficiency in indexing.

### Training Pipeline
Since our dataset already have been preprocessed into passages, all we have to do is extract question/passage pairs from datasets
The model we chose was the 'electra-small-discriminator', it can be thought of as a lite version of BERT. Similarity is computed with dot product.
As for our evaluation metrics, we use 'recall' and Mean Reciprocal Rank (MRR), to measure how our validation set performs after every epoch.
For future work, I will incorporate BERTscore as part of the evaluation metric.
Once a desired result is achieved, the model dict_state will be loaded.

### Quick Guide:
##### 1. Create a .env file:
```
MONGO_USERNAME = {username}
MONGO_PASSWORD = {password}
DATABASE_NAME = Cooking
COLLECTION_NAME = CookEmbedding

DATABASE_NAME_1 = CookingCorpus
COLLECTION_NAME_1 = Passages

MONGO_LINK = {link}
multi_gpu = False
a_name = 'google/electra-small-discriminator'
q_name = 'google/electra-small-discriminator'
t_name = 'google/electra-small-discriminator'
a_path = "model/a_encoder_model.bin"
q_path = "model/q_encoder_model.bin"
```
The ones wrapped around bracket are placeholder values, but for the connection to the mongodb cloud, it should have the following format:
```
"mongodb+srv://{username}:{password}@{link}"
```
multi_gpu: If you have multiple GPU on, change it to **True**

#### 2. Database Injection:
Ensure that the connection to the mongodb database is fine. Then run all the command in **database_injection.ipynb** to inject the necessary embeddings, and the answer passage to the database.

#### 3. Run the Inference!
Finally, if everything has been correctly set up. Go to _inference.py_ and run the inference!

```
title3 = "Roasting a Juicy Turkey"
body3 = "Thanksgiving is coming up and I'm in charge of the turkey this year. I always worry it will be too dry. What's the best way to roast a turkey so it stays moist and flavorful?"
```
If you want to change the inputs, you can tinker with the variables above. 
For the following input in the above (since we set k (number of passages retrieved) to 5, there will be 5 passages in our output):

```
['I agree with Nick if you will be carving the turkey at the table. For Thanksgiving, I always roast two turkeys at a time. One is pretty and displayed and carved. The other I carve before the meal and use in case the first runs out we have big groups and for left overs. I always use the roasting oven for the non display turkey. I have used it from time to time for the display turkey as well after roasting in the oven as Nick said it doesn t brown as well but it is ok. ', 'Personally, I really enjoy turkey in the roaster oven. We purchased ours for a Thanksgiving camping trip and I ve never gone back to the old oven again. The trick is to brown up the skin to begin with. Make sure to coat the skin with olive oil, then the turkey goes in the roaster for 30 minutes at 500 degrees. That browns up the skin to begin with. After 30 minutes, turn the heat down to 325 degrees and keep cooking for the allotted amount of time. DO NOT REMOVE THE LID! This makes a perfect turkey every time. ', 'I like to practice too if it s something I ve never made before. Roasting a chicken will give you an opportunity to practice the all important skill of correctly placing the thermometer, but chicken has a different flavor than turkey. One option is to get the smallest turkey you can find, experiment with a brine which I highly recommend for roasted turkey and perhaps go into Thanksgiving with a bit more confidence. You can always cut up the turkey meat and freeze it for sometime after Thanksgiving when you re no longer sick of turkey. Leftover turkey makes great tamales or enchiladas. ', 'The purpose as with any cooked meat is to let the meat firm up so it doesn t release juices when you cut into it. In the case of a turkey it also helps to let it cool enough to not burn you when you are carving and eating it. Both of these goals will be met in 30 minutes to an hour. I don t know why that chef would recommend 3 hours. At that length of time the turkey would start to approach room temperature and would be less appealing to eat as well as start the clock on the danger zone. ', 'I work in a deli and over the holidays we cook and debone a lot of turkeys. I can tell you from experience that a 20 turkey will yield about 8 10 pounds of meat. This is because when the turkey cooks it releases a surprising amount of liquid. Add in all the bones and you loose a lot of your starting weight. So you really do want to figure 1 1 2 to 2 pounds per person, especially if you want leftovers. ']
```
###### Now finally you can play around with your own DPR based Open Domain Question Answer System!
#### Next steps:
Make an MVP with the backend, then try to build an API that can be run serverless.



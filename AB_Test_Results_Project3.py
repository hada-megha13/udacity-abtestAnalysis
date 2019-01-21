
# coding: utf-8

# ## Analyze A/B Test Results
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# In[2]:


import os
os.getcwd()


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[4]:


df= pd.read_csv("ab_data.csv")
df.head(5)


# b. Use the below cell to find the number of rows in the dataset.

# In[5]:


df.shape[0]


# c. The number of unique users in the dataset.

# In[6]:


len(df.user_id.unique())


# d. The proportion of users converted.

# In[7]:


df['converted'].mean()


# e. The number of times the `new_page` and `treatment` don't line up.

# In[8]:


len(df[(df["group"] != "treatment") & (df["landing_page"] != "new_page")])


# f. Do any of the rows have missing values?

# In[14]:


df.isnull().values.ravel().sum()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# #### Dropping the rows which do not have treatment aligned with new page and control aligned with old page
# 

# In[15]:


df2=df[~((df["group"] == "treatment") & (df["landing_page"] == "old_page"))]
df2=df2[~((df2["group"] == "control") & (df2["landing_page"] == "new_page"))]
df2.shape[0]


# #### Double Check all of the correct rows were removed - this should be 0

# In[16]:


df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[18]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[19]:


df2[df2.user_id.duplicated()]


# c. What is the row information for the repeat **user_id**? 

# In[ ]:


df2[df2.user_id.duplicated()]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[20]:


df2.drop_duplicates(subset="user_id", keep="first", inplace=True)
df2.shape[0]


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[21]:


df2.converted.value_counts()[1]/len(df2.user_id.unique())


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[23]:


len(df2[((df2["group"] == "control") & (df2["converted"] == 1))])/len(df2[df2["group"]=="control"])


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[24]:


len(df2[((df2["group"] == "treatment") & (df2["converted"] == 1))])/len(df2[df2["group"]=="treatment"])


# d. What is the probability that an individual received the new page?

# In[25]:


len(df2[df2["landing_page"]=="new_page"])/df2.shape[0]


# e. Consider your results from a. through d. above, and explain below whether you think there is sufficient evidence to say that the new treatment page leads to more conversions.

# ## Answer
# 
# From the above analysis, it has been observed that the probability of an individual being converted when in treatment group is 0.118 and for those who are in control group is  0.120. Both probalities are very close to each other. Thus, it does not provide sufficient evidence to say that the new treatment page leads to more conversions.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# ## Hypothesis
# 
# $$H_0: p_{old}=p_{new}=0.119$$
# 
# 
# $$H_0: p_{old}<p_{new} $$

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# In[32]:


### taking sample size as same as one in the dataframe
sample1=df2.sample(len(df2[df2["landing_page"]=="old_page"]), replace=True)
sample2=df2.sample(len(df2[df2["landing_page"]=="new_page"]), replace=True)


# In[36]:


sample1.head()


# In[40]:


### simulate sampling distribution for old page and new page
old_pg_conv, new_pg_conv = [], []

for _ in range(10000):
    old_pg = len(sample1[((sample1["landing_page"] == "old_page") & (sample1["converted"] == 1))])/len(sample1[sample1["landing_page"]=="old_page"])
    new_pg =len(sample2[((sample2["landing_page"] == "new_page") & (sample2["converted"] == 1))])/len(sample2[sample2["landing_page"]=="new_page"])
    # append the info 
    old_pg_conv.append(old_pg)
    new_pg_conv.append(new_pg)
    


# #### Simulating under null

# In[50]:


#### simulating under the null for new page
### Setting the null mean to the converted rate in dataframe
null_mean=df2.converted.value_counts()[1]/len(df2.user_id.unique())
std_new_pg=np.std(new_pg_conv)

new_pg_null = np.random.normal(null_mean, std_new_pg, 10000)


# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[51]:


new_pg_null.mean()


# In[ ]:


#### simulating under the null for old page
### Setting the null mean to the converted rate in dataframe
null_mean=df2.converted.value_counts()[1]/len(df2.user_id.unique())
std_old_pg=np.std(old_pg_conv)

old_pg_null = np.random.normal(null_mean, std_old_pg, 10000)


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[52]:


old_pg_null.mean()


# c. What is $n_{new}$?

# In[54]:


sample2.shape[0]


# d. What is $n_{old}$?

# In[55]:


sample1.shape[0]


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[62]:


null_mean=df2.converted.value_counts()[1]/len(df2.user_id.unique())
std_new_pg=np.std(new_pg_null)

new_pg_2 = np.random.normal(null_mean, std_new_pg, sample2.shape[0])


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[57]:


null_mean=df2.converted.value_counts()[1]/len(df2.user_id.unique())
std_old_pg=np.std(old_pg_null)

new_pg_1 = np.random.normal(null_mean, std_old_pg, sample1.shape[0])


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[60]:


p_new=new_pg_2.mean()
p_old=new_pg_1.mean()

p_diff=p_new-p_old
p_diff


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in a numpy array called **p_diffs**.

# In[66]:



p_diffs = []
size = df2.shape[0]
for _ in range(10000):
    b_samp = df2.sample(size, replace=True)
    old_pg_df = b_samp.query('landing_page == "old_page"')
    new_pg_df = b_samp.query('landing_page == "new_page"')
    p_old = old_pg_df['converted'].mean()
    p_new = new_pg_df['converted'].mean()
    p_diffs.append(p_new - p_old)


# In[67]:


# convert to numpy array
p_diffs = np.array(p_diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[68]:


plt.hist(p_diffs)


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[70]:


# create distribution under the null hypothesis
null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)

# observed diffs
p_old_obs = len(df2[((df2["landing_page"] == "old_page") & (df2["converted"] == 1))])/len(df2[df2["landing_page"]=="old_page"])
p_new_obs = len(df2[((df2["landing_page"] == "new_page") & (df2["converted"] == 1))])/len(df2[df2["landing_page"]=="new_page"])

obs_diff = p_new_obs-p_old_obs

# compute p value
(null_vals > obs_diff).mean()


# k. In words, explain what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# ### Answer to k-
# In part k, we have calculated the proportion of values of p_diffs which are greater than the actual difference observed in the given data. The value computed above is called as p-value in scientific studies. Since the above computed p-value is greater than alpha value of 0.05, we have insufficient evidence to reject the null hypothesis. Thus, we fail to reject the null hypothesis which says that there is no significant difference between the conversion rates of old page and new page.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[78]:


convert_old = len(df2[((df2['landing_page'] == "old_page") & (df2["converted"] == 1))])
convert_new = len(df2[((df2['landing_page'] == "new_page") & (df2["converted"] == 1))])
n_old = len(df2[df2['landing_page']=='old_page'])
n_new = len(df2[df2['landing_page']=='new_page'])


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[80]:


import statsmodels.api as sm
z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new])
print(z_score)
print(p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# ## Answer to n-
# 
# The above z score tells us that the value of standard deviation from the mean in given data observations. Also, as per the p value computed in section m, we again fail to reject the null hypothesis of no significant difference between the conversions done by old and new page. Yes, these findings do agree with the findings in part j and k.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# ### Answer
# 
# Since, here each row is either conversion or no conversion, it is clearly a problem of classification in which we would like to classify an individual being converted or not converted

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[93]:


import statsmodels.api as sm

df2['intercept'] = 1
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']
df2[['old_page','new_page']] = pd.get_dummies(df2['landing_page'])
df2.head()


# ## How long did this test run?

# In[111]:


print(df2.timestamp.max())
print(df2.timestamp.min())


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[94]:


log_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page', 'new_page']])
results = log_mod.fit()
results.summary()


# ### Removing new_page variable from the model
# As, group and landing page are related to one another, it could be a good exercise to remove one of them.

# In[97]:


log_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = log_mod.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[96]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# ## Answer-
# The p-value associated with ab_page is 0.19. It differs from the p value found in Part II because this p-value tests the null hypothesis of there is no effect of ab_page(group) on conversion of an individual, whereas p-Value in Part II is associated with testing of null hypothesis of there is no significant difference in  the conversion from old page to new page.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# ## Answer-
# Since, from the above model, it has been observed that the predictor variable ab_page is not significant for our dependent variable "converted". Thus, it could be a better approach to add some other variables that may influence the conversion. However, while introducing more predictor variables into the model, we have to be cautious about multi-collinearity between the predictor variables.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[102]:


countries_df = pd.read_csv('countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.country.unique()


# In[104]:


### Create the necessary dummy variables
country_dummies = pd.get_dummies(df_new['country'])
df_new2 = df_new.join(country_dummies)
df_new2.head()


# In[108]:


log_mod = sm.Logit(df_new2['converted'], df_new2[['intercept', 'CA', 'UK']])
results = log_mod.fit()
results.summary()


# ## Observation from Model including only Country Variable
# As per the p-value, country does not seem to have a significant impact on conversion. 

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# ## Interaction between Page and Country

# In[109]:


log_mod = sm.Logit(df_new2['converted'], df_new2[['intercept', 'ab_page', 'CA', 'UK']])
results = log_mod.fit()
results.summary()


# <a id='conclusions'></a>
# ## Conclusions
# 
# ### Summary
# In the above project, I have worked on two different approaches to understand the results of an A/B test run by an e-commerce website. Below are the observations that can be made from the project-
# 
# 1. From hypothesis testing, it has been observed that there is no significant difference between the conversion rate by old page and new page. 
# 2. It has been observed from the data that A/B test run by the website was for a very short span of time(23 days), which is not sufficient to understand the pattern or behaviour of individuals towards the change.
# 3. From the regression approach, it has been observed that there is no significant impact of change in page on the conversion.
# 
# ### Conclusions
# 
# From the above results obtained from hypothsis testing, it can be concluded that the company should perhaps run the experiment longer to make their decision. Since, from the above results it is clearly seen that new page implementation would not lead to more conversions as compared to the old page.

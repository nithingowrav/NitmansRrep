
# coding: utf-8

# In[2]:


import json
import pandas as pd

with open(r'Urls_data.txt',encoding="utf8") as json_file:
    data = [json.loads(line) for line in json_file]
    data_sub=pd.DataFrame(data)


# In[3]:


data_sub.columns


# In[4]:


cust_data= pd.DataFrame(list(open('part-00000', 'r')))


# In[66]:


cust_data.shape


# In[6]:


cust_data[0] = cust_data[0].str.replace('\n', '')


# In[7]:


cust_url = cust_data[0].str.split(",", n = 1, expand = True) 


# In[8]:


cust_url=cust_url.rename(columns=cust_url.iloc[0]).drop(cust_url.index[0])


# In[9]:


cust_url.columns


# In[10]:


merge_cust_url=pd.merge(left=cust_url,right=data_sub,left_on='url',right_on='link',how='inner')


# In[11]:


merge_cust_url.sample(5)


# In[15]:


user_id_list.dtype


# In[13]:


user_id_gnder_train = pd.read_csv('UserIdToGenderTrain.csv')


# In[15]:


usr_id_gndr_train=user_id_gnder_train.astype(object)


# In[16]:


usr_id_gndr_train.sample(5)


# In[17]:


merge_cust_url.columns


# In[18]:


usr_id_gndr_train.columns


# In[23]:


cust_url_fact=pd.merge(left=merge_cust_url,right=usr_id_gndr_train,left_on='userid',right_on='userid',how='left')


# In[24]:


cust_url_fact.sample(5)


# In[32]:


brnd_lst=cust_url_fact['brand'].unique()


# In[ ]:


(tuple(cust_url_fact['tags']))


# In[41]:


unique_tag = []
for tup in cust_url_fact['tags']:
    if tup not in unique_tag:
        unique_tag.append(tup)


# In[43]:


unique_entities=[]
for tup in cust_url_fact['entities']:
    if tup not in unique_entities:
        unique_entities.append(tup)


# In[44]:


pwd


# In[132]:


import os
file_list=list((os.listdir('/resources/Part_file_src')))


# In[133]:


file_list


# In[128]:


prt_file_lst=list(file_list[file_list[0].str.contains("part")])


# In[126]:


prt_file_lst


# In[122]:


prt_files=test.drop(columns='index')


# In[123]:


prt_files


# In[142]:


for i in file_list:
    tgt_cust_url=[]
    stg_cust_url=pd.DataFrame(list(open(i, 'r')))
    tgt_cust_url=pd.concat([tgt_cust_url,stg_cust_url])


# In[91]:


i='part-00011'
test=pd.DataFrame(list(open(i,'r')))


# In[140]:


stg_cust_url.shape


# In[139]:


tgt_cust_url.shape


# In[ ]:


cust_data[0] = cust_data[0].str.replace('\n', '')


# In[16]:


merged_cust_url[merged_cust_url['userid']=='1029643']


# ****sample to unstack a dataframe --> cust_url_unpacked=cust_url_fact.unstack().apply(pd.Series)

# In[20]:


male_cust_fact=cust_url_fact[cust_url_fact['gender']=='M']


# In[21]:


female_cust_fact=cust_url_fact.where(cust_url_fact['gender']=='F')


# In[ ]:


df_lists.plot.bar(rot=0, cmap=plt.cm.jet, fontsize=8, width=0.7, figsize=(8,4))


# In[20]:


import numpy as np
import matplotlib.pyplot as plt


# In[22]:


# Plot
plt.bar(cust_url_fact['tags'],cust_url_fact['userid'].count(), alpha=0.5, label=cust_url_fact['gender'])
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


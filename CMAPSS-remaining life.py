#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd

# load dataset from local folder
train_df = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None)
test_df = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None)
test_rul_df = pd.read_csv("RUL_FD001.txt", header=None)

# correct column names
columns = ['engine_number','time_in_cycles'] \
+ [f'op_setting_{i}' for i in range(1,4)] \
+ [f'sensor_{i}' for i in range(1,22)]

train_df.columns = columns
test_df.columns = columns

# reset index only
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# In[47]:


train_df


# In[48]:


test_df


# In[4]:


# we remove the specified columns
columns_to_remove = ['op_setting_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_19']
train_df.drop(columns=columns_to_remove, inplace=True)
test_df.drop(columns=columns_to_remove, inplace=True)


# In[5]:


train_df


# In[6]:


test_data = pd.read_csv("test_FD001.txt", sep = "\s+", header = None)
true_rul = pd.read_csv("RUL_FD001.txt", sep = '\s+', header = None)
train_data = pd.read_csv("train_FD001.txt", sep = "\s+", header = None)



# In[7]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# LOAD DATA
test_df = pd.read_csv("test_FD001.txt", sep="\s+", engine="python", header=None)
train_df = pd.read_csv("train_FD001.txt", sep="\s+", engine="python", header=None)

# REMOVE EMPTY COLUMNS
test_df = test_df.dropna(axis=1)
train_df = train_df.dropna(axis=1)

# DEFINE COLUMNS
columns = ['engine_number', 'time_in_cycles'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

# ASSIGN COLUMN NAMES (only works if 26 columns)
test_df.columns = columns
train_df.columns = columns

# REMOVE THESE (they were wrong — deleting real data)
# test_df = test_df.drop(index=0).reset_index(drop=True)
# train_df = train_df.drop(index=0).reset_index(drop=True)


# In[8]:


test_df


# In[9]:


engine_time_df = test_df[['engine_number', 'time_in_cycles']].copy()

columns_to_be_dropped = [
    'engine_number','time_in_cycles',
    'op_setting_3','sensor_1','sensor_5',
    'sensor_10','sensor_16','sensor_19'
]

test_df_dropped = test_df.drop(columns=columns_to_be_dropped)

scaler = MinMaxScaler()

test_df_scaled = scaler.fit_transform(test_df_dropped.astype(float))

test_df_scaled_df = pd.DataFrame(
    test_df_scaled,
    columns=test_df_dropped.columns
)

test_df_scaled_df.to_csv('test_FD001_normalized.csv', header=False, index=False)


# In[10]:


test_df_scaled_df


# In[11]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# LOAD
train_df = pd.read_csv("train_FD001.txt", sep="\s+", engine="python", header=None)

# CLEAN
train_df = train_df.dropna(axis=1)

# COLUMNS
columns = ['engine_number', 'time_in_cycles'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

train_df.columns = columns

# DROP UNUSED
columns_to_be_dropped = [
    'engine_number','time_in_cycles',
    'op_setting_3','sensor_1','sensor_5',
    'sensor_10','sensor_16','sensor_19'
]

train_df_dropped = train_df.drop(columns=columns_to_be_dropped)

# SCALE
scaler = MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df_dropped.astype(float))

# DATAFRAME
train_df_scaled_df = pd.DataFrame(
    train_df_scaled,
    columns=train_df_dropped.columns
)


# In[12]:


engine_time_df


# In[13]:


from sklearn.preprocessing import MinMaxScaler

features_to_normalize = train_df.columns[2:]

train_df[features_to_normalize] = train_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')
test_df[features_to_normalize] = test_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')

# we use the min-max-scaler in sklearn lib
scaler = MinMaxScaler()
train_df[features_to_normalize] = scaler.fit_transform(train_df[features_to_normalize])
test_df[features_to_normalize] = scaler.transform(test_df[features_to_normalize])


# In[14]:


train_df = pd.read_csv('train_FD001.txt', sep="\s+", header=None)

columns = ['engine_number', 'time_in_cycles'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
train_df.columns = columns

max_cycles = train_df.groupby('engine_number')['time_in_cycles'].max().reset_index()

sorted_max_cycles = max_cycles.sort_values(by='engine_number').reset_index(drop=True)

print(sorted_max_cycles)


# In[15]:


train_df['time_in_cycles'] = pd.to_numeric(train_df['time_in_cycles'], errors='coerce')

rul = pd.DataFrame(train_df.groupby('engine_number')['time_in_cycles'].max()).reset_index()
rul.columns = ['engine_number', 'max_time_in_cycles']
train_df = train_df.merge(rul, on='engine_number')

train_df['RUL'] = train_df['max_time_in_cycles'] - train_df['time_in_cycles']
train_df.drop('max_time_in_cycles', axis=1, inplace=True)

train_df['RUL'] = train_df['RUL'].apply(lambda x: min(x, 130))

time_window = 50
train_df['label'] = train_df['RUL'].apply(lambda x: 1 if x > time_window else 0)


# In[16]:


train_df


# In[17]:


train_rul = pd.DataFrame(train_df.groupby('engine_number')['time_in_cycles'].max()).reset_index()
train_rul


# In[18]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# FIX LOADING (critical)
train_df = pd.read_csv('train_FD001.csv', sep=r'\s+|,', engine='python', header=None)
test_df = pd.read_csv('test_FD001.csv', sep=r'\s+|,', engine='python', header=None)
test_rul_df = pd.read_csv('RUL_FD001.csv', header=None)

# DEFINE COLUMNS (match actual shape)
if train_df.shape[1] == 26:
    columns = ['engine_number', 'time_in_cycles'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
elif train_df.shape[1] == 27:
    columns = ['rows', 'engine_number', 'time_in_cycles'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
else:
    raise ValueError("Wrong column count")

train_df.columns = columns
test_df.columns = columns

# REMOVE WRONG ROW DROPS (these were corrupting output)
# test_df = test_df.drop(index=0)
# train_df = train_df.drop(index=0)

# DROP UNUSED
columns_to_remove = ['op_setting_3', 'sensor_1', 'sensor_5',
                     'sensor_10', 'sensor_16', 'sensor_19']

if 'rows' in train_df.columns:
    columns_to_remove.append('rows')

train_df.drop(columns=columns_to_remove, inplace=True)
test_df.drop(columns=columns_to_remove, inplace=True)

# NORMALIZE
features_to_normalize = train_df.columns[2:]

train_df[features_to_normalize] = train_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')
test_df[features_to_normalize] = test_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')

scaler = MinMaxScaler()
train_df[features_to_normalize] = scaler.fit_transform(train_df[features_to_normalize])
test_df[features_to_normalize] = scaler.transform(test_df[features_to_normalize])

# RUL
train_df['time_in_cycles'] = pd.to_numeric(train_df['time_in_cycles'], errors='coerce')

rul = train_df.groupby('engine_number')['time_in_cycles'].max().reset_index()
rul.columns = ['engine_number', 'max_time_in_cycles']

train_df = train_df.merge(rul, on='engine_number')

train_df['RUL'] = train_df['max_time_in_cycles'] - train_df['time_in_cycles']
train_df.drop('max_time_in_cycles', axis=1, inplace=True)

train_df['RUL'] = train_df['RUL'].apply(lambda x: min(x, 130))

time_window = 50
train_df['label'] = train_df['RUL'].apply(lambda x: 1 if x > time_window else 0)

# FINAL OUTPUT (this matches your expected format)
train_rul = train_df.groupby('engine_number')['time_in_cycles'].max().reset_index()

print(train_rul)


# In[19]:


train_df


# In[20]:


train_df.to_csv('train_FD001_complete.csv', header=False, index=False)


# In[21]:


train_df['RUL'][0:192]


# In[22]:


# Plotting the first 192 RUL values
import numpy as np
import matplotlib.pyplot as plt
x_range = np.linspace(1, 193, 192)
plt.plot(x_range, train_df['RUL'][0:192])
plt.xlabel('Index')
plt.ylabel('RUL')
plt.title('RUL over Index Range')
plt.show()


# In[23]:


#for regression
def create_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1, -2])
    return np.array(X), np.array(y)

window_size = 30
X_train_r, y_train_r = create_time_windows(train_df, window_size, 1)

print(X_train_r.shape, y_train_r.shape)


# In[24]:


X_train_r[0,1,:]


# In[25]:


y_train_r


# In[26]:


# for classification
def create_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1, -1])  # thats for classification!
    return np.array(X), np.array(y)

window_size = 30
X_train_c, y_train_c = create_time_windows(train_df, window_size, 1)

print(X_train_c.shape, y_train_c.shape)


# In[27]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv('train_FD001.csv', sep=r'\s+|,', engine='python', header=None)
test_df = pd.read_csv('test_FD001.csv', sep=r'\s+|,', engine='python', header=None)
test_rul_df = pd.read_csv('RUL_FD001.csv', header=None)


if train_df.shape[1] == 26:
    columns = ['engine_number', 'time_in_cycles'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
elif train_df.shape[1] == 27:
    columns = ['rows', 'engine_number', 'time_in_cycles'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
else:
    raise ValueError(f"Wrong column count: {train_df.shape[1]}")

train_df.columns = columns
test_df.columns = columns

columns_to_remove = ['op_setting_3', 'sensor_1', 'sensor_5',
                     'sensor_10', 'sensor_16', 'sensor_19']

if 'rows' in train_df.columns:
    columns_to_remove.append('rows')

train_df.drop(columns=columns_to_remove, inplace=True)
test_df.drop(columns=columns_to_remove, inplace=True)

features_to_normalize = train_df.columns[2:]

train_df[features_to_normalize] = train_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')
test_df[features_to_normalize] = test_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')

scaler = MinMaxScaler()
train_df[features_to_normalize] = scaler.fit_transform(train_df[features_to_normalize])
test_df[features_to_normalize] = scaler.transform(test_df[features_to_normalize])

train_df['time_in_cycles'] = pd.to_numeric(train_df['time_in_cycles'], errors='coerce')

rul = train_df.groupby('engine_number')['time_in_cycles'].max().reset_index()
rul.columns = ['engine_number', 'max_time_in_cycles']

train_df = train_df.merge(rul, on='engine_number')

train_df['RUL'] = train_df['max_time_in_cycles'] - train_df['time_in_cycles']
train_df.drop('max_time_in_cycles', axis=1, inplace=True)

train_df['RUL'] = train_df['RUL'].apply(lambda x: min(x, 130))

time_window = 50
train_df['label'] = train_df['RUL'].apply(lambda x: 1 if x > time_window else 0)

train_rul = train_df.groupby('engine_number')['time_in_cycles'].max().reset_index()

def create_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1, -2])
    return np.array(X), np.array(y)

window_size = 30
X_train_r, y_train_r = create_time_windows(train_df, window_size, 1)

print(X_train_r.shape, y_train_r.shape)


# In[28]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

test_df = pd.read_csv('test_FD001.csv', sep=r'\s+|,', engine='python', header=None)
test_rul_df = pd.read_csv('RUL_FD001.csv', header=None)

if test_df.shape[1] == 26:
    columns = ['engine_number', 'time_in_cycles'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
elif test_df.shape[1] == 27:
    columns = ['rows', 'engine_number', 'time_in_cycles'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
else:
    raise ValueError("Wrong column count")

test_df.columns = columns

cols_remove = ['op_setting_3', 'sensor_1', 'sensor_5',
               'sensor_10', 'sensor_16', 'sensor_19']

if 'rows' in test_df.columns:
    cols_remove.append('rows')

test_df.drop(columns=cols_remove, inplace=True)

features = test_df.columns[2:]
test_df[features] = test_df[features].apply(pd.to_numeric, errors='coerce')

scaler = MinMaxScaler()
test_df[features] = scaler.fit_transform(test_df[features])

test_df['time_in_cycles'] = pd.to_numeric(test_df['time_in_cycles'], errors='coerce')

rul = test_df.groupby('engine_number')['time_in_cycles'].max().reset_index()
rul.columns = ['engine_number', 'max_time_in_cycles']

test_df = test_df.merge(rul, on='engine_number')

test_df['RUL'] = test_df['max_time_in_cycles'] - test_df['time_in_cycles']

test_rul_df.columns = ['RUL']
test_rul_df['engine_number'] = test_rul_df.index + 1

test_df = test_df.merge(test_rul_df, on='engine_number', how='left')

test_df['RUL_total'] = test_df['RUL_x'] + test_df['RUL_y']

print(test_df.head(60))


# In[29]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

test_df = pd.read_csv('test_FD001.csv', sep=r'\s+|,', engine='python', header=None)
test_rul_df = pd.read_csv('RUL_FD001.csv', header=None)

columns = ['engine_number', 'time_in_cycles'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

test_df.columns = columns

columns_to_remove = ['op_setting_3', 'sensor_1', 'sensor_5',
                     'sensor_10', 'sensor_16', 'sensor_19']
test_df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

features_to_normalize = test_df.columns[2:]
test_df[features_to_normalize] = test_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')

scaler = MinMaxScaler()
test_df[features_to_normalize] = scaler.fit_transform(test_df[features_to_normalize])

test_df['time_in_cycles'] = pd.to_numeric(test_df['time_in_cycles'], errors='coerce')

rul = test_df.groupby('engine_number')['time_in_cycles'].max().reset_index()
rul.columns = ['engine_number', 'max_time_in_cycles']

test_df = test_df.merge(rul, on='engine_number')

test_df['RUL'] = test_df['max_time_in_cycles'] - test_df['time_in_cycles']
test_df.drop('max_time_in_cycles', axis=1, inplace=True)

test_rul_df.columns = ['RUL']
test_rul_df['unit_number'] = test_rul_df.index + 1
test_rul_df['RUL'] = pd.to_numeric(test_rul_df['RUL'], errors='coerce')

rul = rul.merge(test_rul_df[['unit_number', 'RUL']], left_on='engine_number', right_on='unit_number', how='left')

test_df = test_df.merge(rul[['engine_number', 'max_time_in_cycles', 'RUL']], on='engine_number')

test_df['RUL'] = test_df['RUL_x'] + test_df['RUL_y']
test_df['RUL'] = test_df['RUL'].apply(lambda x: min(x, 130))
test_df.drop(columns=['RUL_x', 'max_time_in_cycles', 'RUL_y'], inplace=True)

time_window = 50
test_df['label'] = test_df['RUL'].apply(lambda x: 1 if x > time_window else 0)

test_rul = test_df.groupby('engine_number')['time_in_cycles'].max().reset_index()

def create_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1, -2])
    return np.array(X), np.array(y)

window_size = 30
X_test_r, y_test_r = create_time_windows(test_df, window_size, 1)

print(X_test_r.shape, y_test_r.shape)


# In[30]:


y_test_r[0:200]


# In[31]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# LOAD DATA (CRITICAL FIX: whitespace separator)
# ---------------------------
train_df = pd.read_csv('train_FD001.csv', sep=r'\s+', header=None)
test_df = pd.read_csv('test_FD001.csv', sep=r'\s+', header=None)
test_rul_df = pd.read_csv('RUL_FD001.csv', sep=r'\s+', header=None)

# ---------------------------
# CORRECT COLUMN NAMES (26 columns total)
# ---------------------------
columns = ['engine_number', 'time_in_cycles'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

train_df.columns = columns
test_df.columns = columns

# RUL file has only 1 column
test_rul_df.columns = ['RUL']

# ---------------------------
# REMOVE UNUSED FEATURES
# ---------------------------
columns_to_remove = ['op_setting_3', 'sensor_1', 'sensor_5', 
                     'sensor_10', 'sensor_16', 'sensor_19']

train_df.drop(columns=columns_to_remove, inplace=True)
test_df.drop(columns=columns_to_remove, inplace=True)

# ---------------------------
# NORMALIZATION
# ---------------------------
features_to_normalize = train_df.columns[2:]

train_df[features_to_normalize] = train_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')
test_df[features_to_normalize] = test_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')

scaler = MinMaxScaler()
train_df[features_to_normalize] = scaler.fit_transform(train_df[features_to_normalize])
test_df[features_to_normalize] = scaler.transform(test_df[features_to_normalize])

# ---------------------------
# COMPUTE RUL FOR TEST SET
# ---------------------------
test_df['time_in_cycles'] = pd.to_numeric(test_df['time_in_cycles'], errors='coerce')

rul = test_df.groupby('engine_number')['time_in_cycles'].max().reset_index()
rul.columns = ['engine_number', 'max_time_in_cycles']

test_df = test_df.merge(rul, on='engine_number')

test_df['RUL'] = test_df['max_time_in_cycles'] - test_df['time_in_cycles']
test_df.drop('max_time_in_cycles', axis=1, inplace=True)

# Add final RUL from file
test_rul_df['unit_number'] = test_rul_df.index + 1

rul = rul.merge(test_rul_df, left_on='engine_number', right_on='unit_number', how='left')

test_df = test_df.merge(rul[['engine_number', 'max_time_in_cycles', 'RUL']], on='engine_number')

test_df['RUL'] = test_df['RUL_x'] + test_df['RUL_y']
test_df['RUL'] = test_df['RUL'].apply(lambda x: min(x, 130))

test_df.drop(columns=['RUL_x', 'RUL_y', 'max_time_in_cycles'], inplace=True)

# ---------------------------
# LABELING
# ---------------------------
time_window = 50
test_df['label'] = test_df['RUL'].apply(lambda x: 1 if x > time_window else 0)

# ---------------------------
# CREATE TIME WINDOWS
# ---------------------------
def create_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(len(engine_df) - window_size, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1]['RUL'])
    return np.array(X), np.array(y)

window_size = 30
X_test_r, y_test_r = create_time_windows(test_df, window_size, 1)

# ---------------------------
# FINAL OUTPUT
# ---------------------------
print(X_test_r.shape, y_test_r.shape)


# In[32]:


y_test_r


# In[33]:


test_df = pd.read_csv('test_FD001.csv', sep=r'\s+', header=None)

columns = ['engine_number', 'time_in_cycles'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

test_df.columns = columns


# In[34]:


# max cycle per engine
max_cycle_test = test_df.groupby('engine_number')['time_in_cycles'].max().reset_index()
max_cycle_test.columns = ['engine_number', 'max_time_in_cycles']

# fix RUL file
test_rul_df = pd.read_csv('RUL_FD001.csv', sep=r'\s+', header=None)

if test_rul_df.shape[1] > 1:
    test_rul_df = test_rul_df.iloc[:, 0]

test_rul_df = pd.DataFrame(test_rul_df)
test_rul_df.columns = ['RUL']
test_rul_df['engine_number'] = test_rul_df.index + 1

# merge RUL into max_cycle
max_cycle_test = max_cycle_test.merge(test_rul_df, on='engine_number', how='left')

# merge into test_df (FIRST TIME ONLY)
test_df = test_df.merge(max_cycle_test, on='engine_number', how='left')

# compute final RUL
test_df['RUL'] = test_df['max_time_in_cycles'] + test_df['RUL'] - test_df['time_in_cycles']


# In[35]:


def create_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1]['RUL'])
    return np.array(X), np.array(y)


# In[36]:


def create_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end][feature_cols].values)
            y.append(engine_df.iloc[end - 1]['RUL'])
    return np.array(X), np.array(y)


# In[37]:


def create_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end, 2:-2].values)
            y.append(engine_df.iloc[end - 1, -2])
    return np.array(X), np.array(y)

window_size = 30
X_train_r, y_train_r = create_time_windows(train_df, window_size, 1)

print(X_train_r.shape, y_train_r.shape)


# In[38]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# LOAD DATA (correct parsing)
# ---------------------------
train_df = pd.read_csv('train_FD001.csv', sep=r'\s+', header=None)
test_df = pd.read_csv('test_FD001.csv', sep=r'\s+', header=None)
test_rul_df = pd.read_csv('RUL_FD001.csv', sep=r'\s+', header=None)

# ---------------------------
# SET CORRECT COLUMNS (26)
# ---------------------------
columns = ['engine_number', 'time_in_cycles'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

train_df.columns = columns
test_df.columns = columns

# ---------------------------
# REMOVE UNUSED FEATURES
# ---------------------------
columns_to_remove = ['op_setting_3', 'sensor_1', 'sensor_5',
                     'sensor_10', 'sensor_16', 'sensor_19']

train_df.drop(columns=columns_to_remove, inplace=True)
test_df.drop(columns=columns_to_remove, inplace=True)

# ---------------------------
# NORMALIZATION
# ---------------------------
features_to_normalize = train_df.columns[2:]

train_df[features_to_normalize] = train_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')
test_df[features_to_normalize] = test_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')

scaler = MinMaxScaler()
train_df[features_to_normalize] = scaler.fit_transform(train_df[features_to_normalize])
test_df[features_to_normalize] = scaler.transform(test_df[features_to_normalize])

# =====================================================
# TEST DATA RUL COMPUTATION
# =====================================================

# max cycle
max_cycle_test = test_df.groupby('engine_number')['time_in_cycles'].max().reset_index()
max_cycle_test.columns = ['engine_number', 'max_time_in_cycles']

# fix RUL file
if test_rul_df.shape[1] > 1:
    test_rul_df = test_rul_df.iloc[:, 0]

test_rul_df.columns = ['RUL']
test_rul_df['engine_number'] = test_rul_df.index + 1

# merge
max_cycle_test = max_cycle_test.merge(test_rul_df, on='engine_number', how='left')
test_df = test_df.merge(max_cycle_test, on='engine_number', how='left')

# compute final RUL
test_df['RUL'] = test_df['max_time_in_cycles'] + test_df['RUL'] - test_df['time_in_cycles']
test_df['RUL'] = test_df['RUL'].apply(lambda x: min(x, 130))

# label
time_window = 50
test_df['label'] = test_df['RUL'].apply(lambda x: 1 if x > time_window else 0)

# =====================================================
# TRAIN DATA RUL COMPUTATION
# =====================================================

max_cycle_train = train_df.groupby('engine_number')['time_in_cycles'].max().reset_index()
max_cycle_train.columns = ['engine_number', 'max_time_in_cycles']

train_df = train_df.merge(max_cycle_train, on='engine_number', how='left')

train_df['RUL'] = train_df['max_time_in_cycles'] - train_df['time_in_cycles']
train_df.drop(columns=['max_time_in_cycles'], inplace=True)

train_df['RUL'] = train_df['RUL'].apply(lambda x: min(x, 130))
train_df['label'] = train_df['RUL'].apply(lambda x: 1 if x > time_window else 0)

# =====================================================
# FEATURE SELECTION (safe)
# =====================================================

feature_cols = [col for col in train_df.columns if col not in ['engine_number', 'time_in_cycles', 'RUL', 'label']]

# =====================================================
# WINDOW FUNCTION (correct)
# =====================================================

def create_time_windows(df, window_size, step):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end][feature_cols].values)
            y.append(engine_df.iloc[end - 1]['RUL'])
    return np.array(X), np.array(y)

# =====================================================
# CREATE DATASETS
# =====================================================

window_size = 30

# train windows
X_train_r, y_train_r = create_time_windows(train_df, window_size, 1)

# test windows (last window only)
def create_test_windows(df, window_size):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        start = len(engine_df) - window_size
        end = len(engine_df)
        X.append(engine_df.iloc[start:end][feature_cols].values)
        y.append(engine_df.iloc[end - 1]['RUL'])
    return np.array(X), np.array(y)

X_test_r, y_test_r = create_test_windows(test_df, window_size)

# =====================================================
# OUTPUT
# =====================================================

print("Train:", X_train_r.shape, y_train_r.shape)
print("Test:", X_test_r.shape, y_test_r.shape)


# In[39]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA (CRITICAL)
# =========================
train_df = pd.read_csv('train_FD001.csv', sep=r'\s+', header=None)

# =========================
# SET COLUMNS
# =========================
columns = ['engine_number', 'time_in_cycles'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

train_df.columns = columns

# =========================
# REMOVE UNUSED FEATURES
# =========================
train_df.drop(columns=[
    'op_setting_3', 'sensor_1', 'sensor_5',
    'sensor_10', 'sensor_16', 'sensor_19'
], inplace=True)

# =========================
# NORMALIZATION
# =========================
features = train_df.columns[2:]

train_df[features] = train_df[features].apply(pd.to_numeric, errors='coerce')

scaler = MinMaxScaler()
train_df[features] = scaler.fit_transform(train_df[features])

# =========================
# COMPUTE RUL
# =========================
train_df['time_in_cycles'] = pd.to_numeric(train_df['time_in_cycles'], errors='coerce')

max_cycle = train_df.groupby('engine_number')['time_in_cycles'].max().reset_index()
max_cycle.columns = ['engine_number', 'max_time_in_cycles']

train_df = train_df.merge(max_cycle, on='engine_number', how='left')

train_df['RUL'] = train_df['max_time_in_cycles'] - train_df['time_in_cycles']
train_df.drop(columns=['max_time_in_cycles'], inplace=True)

# cap RUL
train_df['RUL'] = train_df['RUL'].apply(lambda x: min(x, 130))

# =========================
# EXACT 18 FEATURES
# =========================
features_18 = [
    'op_setting_1', 'op_setting_2',
    'sensor_2', 'sensor_3', 'sensor_4',
    'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13',
    'sensor_14', 'sensor_15',
    'sensor_17', 'sensor_18',
    'sensor_20', 'sensor_21'
]

# =========================
# WINDOW FUNCTION
# =========================
def create_time_windows(df, window_size, step, feature_cols):
    X, y = [], []
    for engine in df['engine_number'].unique():
        engine_df = df[df['engine_number'] == engine]
        for start in range(0, len(engine_df) - window_size + 1, step):
            end = start + window_size
            X.append(engine_df.iloc[start:end][feature_cols].values)
            y.append(engine_df.iloc[end - 1]['RUL'])
    return np.array(X), np.array(y)

# =========================
# GENERATE DATA
# =========================
X, y = create_time_windows(train_df, 30, 1, features_18)

# sanity check
print("Before split:", X.shape)   # MUST be (17731, 30, 18)

# =========================
# SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=83
)

# =========================
# OUTPUT
# =========================
print("Processed train data shape: ", X_train.shape)
print("Processed validation data shape: ", X_val.shape)
print("Processed train targets shape: ", y_train.shape)
print("Processed validation targets shape: ", y_val.shape)
processed_train_data =  X_train
processed_val_data = X_val
processed_train_targets = y_train
processed_val_targets = y_val


# In[40]:


import sys
print(sys.executable)


# In[41]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

processed_train_data, processed_val_data, processed_train_targets, processed_val_targets = train_test_split(processed_train_data,
                                                                                                            processed_train_targets,
                                                                                                            test_size = 0.2,
                                                                                                            random_state = 83)


processed_train_data=processed_train_data.astype('float32')
processed_val_data=processed_val_data.astype('float32')
processed_train_targets=processed_train_targets.astype('float32')
processed_val_targets=processed_val_targets.astype('float32')



print("Processed train data shape: ", processed_train_data.shape)
print("Processed validation data shape: ", processed_val_data.shape)
print("Processed train targets shape: ", processed_train_targets.shape)
print("Processed validation targets shape: ", processed_val_targets.shape)


# In[42]:


import sys
print(sys.version)


# In[43]:


get_ipython().system('pip install tensorflow')


# In[44]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten,Dropout
import tensorflow as tf
# Checkking if GPU is available
if tf.config.list_physical_devices('GPU'):
    device_name = tf.test.gpu_device_name()
    print(f"Using GPU: {device_name}")
else:
    print("No GPU found. Using CPU.")

def create_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(LSTM(50, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))  
    return model

input_shape = (processed_train_data.shape[1], processed_train_data.shape[2])
model = create_cnn_lstm_model(input_shape)


from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss')

history = model.fit(processed_train_data, processed_train_targets, validation_data=(processed_val_data, processed_val_targets), epochs=100, batch_size=200, callbacks=[early_stopping, model_checkpoint])


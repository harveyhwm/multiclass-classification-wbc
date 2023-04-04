import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
import gc
import json

from collections import Counter,deque

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

DATA_PATH = 'data'
CATEGORIES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
SPLITS = ['train', 'validation', 'test']
CHANNELS = ['red', 'green', 'blue']

def get_raw_python_from_notebook(notebook,python=None):
    if python is None: python=notebook
    with open(notebook+'.ipynb','r') as f:
        rawpy = json.load(f)
    rawpy = [[] if c['source'] == [] else c['source'] for c in rawpy['cells'] if c['cell_type']=='code']
    for r in rawpy:
        r.extend(['\n','\n'])
    raw = [l for r in rawpy for l in r]
    with open(python+'.py', 'w') as f:
        f.write(''.join(raw))
get_raw_python_from_notebook('multiclass_classification_wbc')

def get_model_size(m):
    if type(m) is str:
        m = tf.keras.models.load_model(m)
    size = 0
    for layer in m.layers:
        weight_byte_heuristic = 13.69 # chosen from varied empirical evidence :)
        for w in layer.weights:
            s = w.numpy().shape
            n = 1
            for val in s:
                n *= val
            size += n
    print('Total Neurons:', f'{size:,}')
    print('Estimated Model Size:', np.round(1e-6*weight_byte_heuristic*size,2),'MB')
    #return size,int(weight_byte_heuristic*size)

def get_file_counts(data_path=DATA_PATH, splits=SPLITS, categories=CATEGORIES):
    dirs = {}
    for j in splits:
        dirs[j] = {}
        for i in categories:
            dirs[j][i] = os.path.join(data_path,j,i)
            print('size of', j, 'directory for', i, ':', len(os.listdir(dirs[j][i])))
        print('TOTAL length of', j, 'set :', sum([len(os.listdir(dirs[j][i])) for i in categories]))

def get_file_paths(data_path=DATA_PATH, splits=SPLITS, categories=CATEGORIES, df=True):
    paths = []
    for j in splits:
        for i in categories:
            d = {'category': i , 'split': j}
            p = os.path.join(data_path,j,i)
            [paths.append(dict(d,**{'file_path':os.path.join(p,f),'file':f})) for f in os.listdir(p) if 'jpeg' in f]
    if df is True: return pd.DataFrame.from_dict(paths)
    return paths

def extract_channels(dataset_path, path, categories=CATEGORIES, splits=SPLITS, channels=CHANNELS):
    filepaths = {}
    for s in splits:
        for c in categories:
            for f in [f for f in os.listdir(os.path.join(dataset_path,s,c)) if 'jpeg' in f]:
                p = os.path.join(dataset_path,s,c,f)
                filepaths[p] = {}
                for r in channels:
                    filepaths[p][r] = os.path.join(dataset_path,'channels',s,c,(r+f))

    img = image.load_img(path)
    dims = np.array(img).shape
    channels = np.reshape(img,(dims[0]*dims[1],3)).transpose()
    (red,green,blue) = [np.reshape(channels[i],(dims[0],dims[1])).astype(float) for i in range(3)]
    gb_mean = (0.5*(blue+green))
    rb_mean = (0.5*(red+blue))
    rg_mean = (0.5*(red+green))
    red_dominance = red-gb_mean
    green_dominance = green-rb_mean
    blue_dominance = blue-rg_mean
    red_dominance *= 255/np.max(red_dominance)
    green_dominance *= 255/np.max(green_dominance)
    blue_dominance *= 255/np.max(blue_dominance)

    for r in ['red','green','blue']:
        try:
            os.mkdir(os.path.join(dataset_path,r))
            #print('A')
        except:
            pass
        for s in splits:
            try:
                os.mkdir(os.path.join(dataset_path,r,s))
                #print('B')
            except:
                pass
            for c in categories:
                try:
                    os.mkdir(os.path.join(dataset_path,r,s,c))
                    #print('C')
                except:
                    pass

    plt.imsave(filepaths[path]['red'], red_dominance, cmap='gray')
    plt.imsave(filepaths[path]['green'], green_dominance, cmap='gray')
    plt.imsave(filepaths[path]['blue'], blue_dominance, cmap='gray')

def calc_performance(model, data, indices=None, names=SPLITS, original_data=None, verbose=False):
    if type(data) != list: data = [data]
    if (original_data is not None) and (type(original_data) != list): original_data = [original_data]
    if original_data is None: original_data = data
    if indices is None: indices = [s for s in range(len(data[0].classes))]
    blank_indices = [s for s in range(np.max(indices)) if s not in indices]

    classes, true_classes, accuracy_tables, accuracy, cm = {}, {}, {}, {}, {}

    for i in names:
        classes[i] = [indices[np.argmax(c)] if len(c)>1 else indices[int(np.round(c))] for c in model.predict(data[names.index(i)], verbose=False)]+blank_indices
        true_classes[i] = [indices[j] for j in data[names.index(i)].classes]+blank_indices
        accuracy_tables[i] = pd.concat([pd.DataFrame(classes[i],columns=['classes']),pd.DataFrame(true_classes[i],columns=['true_classes'])],axis=1)
        accuracy_tables[i]['accuracy'] = accuracy_tables[i].apply(lambda r: 1 if r['true_classes']==r['classes'] else 0,axis=1)
        accuracy[i] = sum(accuracy_tables[i]['accuracy'])/len(accuracy_tables[i])
        cm[i] = pd.DataFrame(confusion_matrix(true_classes[i], classes[i], normalize='true'))
        for b in blank_indices:
            cm[i].drop(b,axis=0,inplace=True)
            cm[i].drop(b,axis=1,inplace=True)

    return {
        'classes': classes,
        'true_classes': true_classes,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

def make_confusion_matrix(d1, d2):
    return pd.DataFrame(confusion_matrix(d1, d2, normalize='true'))

#### MODEL CALLBACKS: reuseable

# stop early if no improvement after 5 epochs
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    mode='max',
    restore_best_weights=True
)

# save the model with the maximum validation accuracy 
checkpoint = ModelCheckpoint(
    'models/classification_wbc1.h5',
    monitor='val_accuracy',
    verbose=1,
    mode='max', 
    save_best_only=True
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy', #'val_loss',
    factor=0.1,
    patience=10,
    mode='max',
    verbose=1
)

lr_scheduler = LearningRateScheduler(lambda epoch: 1e-5 * 10**(1.5*epoch/EPOCHS))

# traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch
# def lr_scheduler(epochs=100, lrs=(1e-5,1e-2)):
#     return LearningRateScheduler(
#         lambda epoch: lrs[0] * 10**(np.log10(lrs[1]/lrs[0])*epoch/epochs)
#     )

training_datagen = ImageDataGenerator(
    rescale=1.0/255.0, # rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_val_datagen = ImageDataGenerator(rescale = 1.0/255.0)

def flow_data(generator, data_path, split,shuffle=True, classes=None, class_mode='categorical'):
    return generator.flow_from_directory(
        os.path.join(data_path,split),
        target_size=(128,128),
        classes=classes,
        class_mode=class_mode,
        batch_size=60,
        shuffle=shuffle
    )

np.random.seed(67) # use a consistent seed so shuffling gives expected results
train_data = flow_data(training_datagen, DATA_PATH, 'train')
validation_data = flow_data(test_val_datagen, DATA_PATH, 'validation')
train_data_unshuffled = flow_data(training_datagen, DATA_PATH, 'train', shuffle=False)
validation_data_unshuffled = flow_data(test_val_datagen, DATA_PATH, 'validation', shuffle=False)
test_data_unshuffled = flow_data(test_val_datagen, DATA_PATH, 'test', shuffle=False)

model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model_1.summary()

# here, we get a proxy for the model size based on the number of neurons.  
get_model_size(model_1)

model_1.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adam(), #'rmsprop',
    metrics = ['accuracy']
)

EPOCHS = 100

history1 = model_1.fit(
    train_data,
    epochs = EPOCHS,
    validation_data = validation_data,
    verbose = 1,
    callbacks = [reduce_lr,lr_scheduler,checkpoint]
)
# model_1.save('models/classification_wbc1.h5')

# retrieve best model saved from checkpoints
model_1 = tf.keras.models.load_model('models/classification_wbc1.h5')

classes_1 = {}
classes_1['train'] = [np.argmax(c) for c in model1.predict(train_data_unshuffled)]
classes_1['validation'] = [np.argmax(c) for c in model1.predict(validation_data_unshuffled)]
classes_1['test'] = [np.argmax(c) for c in model1.predict(test_data_unshuffled)]

# here we recover the classes from the main data - it's important to make sure that they are all unshuffled
true_classes_1 = {}
true_classes_1['train'] = train_data_unshuffled.classes
true_classes_1['validation'] = validation_data_unshuffled.classes
true_classes_1['test'] = test_data_unshuffled.classes

accuracy_tables_1 = {}
accuracy_1 = {}
for j in SPLITS:
    accuracy_tables_1[j] = pd.concat([pd.DataFrame(classes_1[j],columns=['classes']),pd.DataFrame(true_classes_1[j],columns=['true_classes'])],axis=1)
    accuracy_tables_1[j]['accuracy'] = accuracy_tables_1[j].apply(lambda r: 1 if r['true_classes']==r['classes'] else 0,axis=1)
    accuracy_1[j] = sum(accuracy_tables_1[j]['accuracy'])/len(accuracy_tables_1[j])

accuracy_1

for j in SPLITS:
    print()
    print(pd.DataFrame(confusion_matrix(true_classes_1[j],classes_1[j],normalize='true')))

# flow from directory using only the labels 0, 2 and 3

categories_4a = ['EOSINOPHIL', 'MONOCYTE', 'NEUTROPHIL']

train_data_4a = flow_data(training_datagen, DATA_PATH, 'train', classes=categories_4a)
validation_data_4a = flow_data(test_val_datagen, DATA_PATH, 'validation', classes=categories_4a)
train_data_unshuffled_4a = flow_data(training_datagen, DATA_PATH, 'train', shuffle=False, classes=categories_4a)
validation_data_unshuffled_4a = flow_data(test_val_datagen, DATA_PATH, 'validation', shuffle=False, classes=categories_4a)
test_data_unshuffled_4a = flow_data(test_val_datagen,DATA_PATH, 'test', shuffle=False, classes=categories_4a)

# test_grid = pd.DataFrame(np.array([test_data_unshuffled.classes,
#     test_data_unshuffled.labels,
#     test_data_unshuffled.filepaths]).transpose(),
#     columns=['preds','actual','filepath'])
# test_grid

model_4a = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_4a.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(), #'rmsprop',
    metrics=['accuracy']
)

# save the model with the maximum validation accuracy 
checkpoint = ModelCheckpoint(
    'models/classification_wbc4a_best.h5',
    monitor='val_accuracy',
    verbose=0,
    mode='max',
    save_best_only=True
)

EPOCHS = 100

history_4a = model_4a.fit(
    train_data_4a,
    epochs=EPOCHS,
    validation_data=validation_data_4a,
    verbose=1,
    callbacks=[lr_scheduler,reduce_lr,checkpoint] # early_stopping
)

model_4a = tf.keras.models.load_model('models/classification_wbc4a_best.h5')

data_orig = [train_data_unshuffled, validation_data_unshuffled, test_data_unshuffled]
data_4a = [train_data_unshuffled_4a, validation_data_unshuffled_4a, test_data_unshuffled_4a]

model_results_4a = calc_performance(model_4a, data_4a, [0, 2, 3], original_data=data_orig)

model_results_4a['accuracy']

for i in model_results_4a['confusion_matrix']:
    print(model_results_4a['confusion_matrix'][i])
    print()

# flow from directory using only the labels 0,3
categories_4b = ['EOSINOPHIL', 'NEUTROPHIL']

train_data_4b = flow_data(training_datagen, DATA_PATH, 'train', classes=categories_4b, class_mode='binary')
validation_data_4b = flow_data(test_val_datagen, DATA_PATH, 'validation', classes=categories_4b, class_mode='binary')
train_data_unshuffled_4b = flow_data(training_datagen, DATA_PATH, 'train', shuffle=False, classes=categories_4b, class_mode='binary')
validation_data_unshuffled_4b = flow_data(test_val_datagen, DATA_PATH, 'validation', shuffle=False, classes=categories_4b, class_mode='binary')
test_data_unshuffled_4b = flow_data(test_val_datagen, DATA_PATH, 'test', shuffle=False, classes=categories_4b, class_mode='binary')

model_4b = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_4b.compile(
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(), # 'rmsprop',
    metrics = ['accuracy']
)

# save the model with the maximum validation accuracy 
checkpoint = ModelCheckpoint(
    'models/classification_wbc4b_best.h5',
    monitor='val_accuracy',
    verbose=0,
    mode='max', 
    save_best_only=True
)

history_4b = model_4b.fit(
    train_data_4b,
    epochs=EPOCHS,
    validation_data=validation_data_4b,
    verbose=1,
    callbacks=[reduce_lr,lr_scheduler,checkpoint]
)

model_4b = tf.keras.models.load_model('models/classification_wbc4b_best.h5')

model_results_4b = calc_performance(model_4b, [train_data_unshuffled_4b, validation_data_unshuffled_4b, test_data_unshuffled_4b], [0, 3])
model_results_4b['accuracy']

for i in model_results_4b['confusion_matrix']:
    print(model_results_4b['confusion_matrix'][i])
    print()

preds_1 = [np.argmax(p) for p in model_1.predict(test_data_unshuffled, verbose=0)] # classes 0,1,2,3
preds_4a = [np.argmax(p) for p in model_4a.predict(test_data_unshuffled, verbose=0)] # classes 0,2,3 (2 is at index 1)
preds_4b = [int(np.round(p)) for p in model_4b.predict(test_data_unshuffled, verbose=0)] # classes 0,3 (3 is at index 1)

classes_test_combined_4 = [1 if preds_1[i] == 1 else 2 if preds_4a[i] == 1 else 3*preds_4b[i] for i in range(len(preds_1))]
combined_accuracy_4 = np.mean([1 if classes_test_combined_4[i]==test_data_unshuffled.classes[i] else 0 for i in range(len(preds_1))])

combined_accuracy_4

print(make_confusion_matrix(test_data_unshuffled.classes, classes_test_combined_4))

classes_test_combined_2_4 = [0 if classes_test_combined_2[i] == 0 else classes_test_combined_4[i] for i in range(len(preds_1))]
combined_accuracy_2_4 = np.mean([1 if classes_test_combined_2_4[i]==test_data_unshuffled.classes[i] else 0 for i in range(len(preds_1))])

combined_accuracy_2_4

print(make_confusion_matrix(test_data_unshuffled.classes, classes_test_combined_2_4))

file_data = get_file_paths()

def flow_df(generator,file_data,shuffle=True):
    return generator.flow_from_dataframe(
        file_data,
        directory=None,
        x_col='file_path',
        y_col='classes',
        target_size=(128,128),
        classes=None,
        # class_mode='binary',
        batch_size=60,
        shuffle=shuffle
    );

binary_models = []

for c in CATEGORIES:
    file_data['classes'] = file_data['category'].apply(lambda x: '0' if x==c else '1')
    file_data_train = file_data[file_data['split']=='train'].reset_index(drop=True)
    file_data_validation = file_data[file_data['split']=='validation'].reset_index(drop=True)
    file_data_test = file_data[file_data['split']=='test'].reset_index(drop=True)

    train_data = flow_df(training_datagen,file_data_train);
    validation_data = flow_df(test_val_datagen,file_data_validation);
    train_data_unshuffled = flow_df(training_datagen,file_data_train,shuffle=False);
    validation_data_unshuffled = flow_df(test_val_datagen,file_data_validation,shuffle=False);
    test_data_unshuffled = flow_df(test_val_datagen,file_data_test,shuffle=False);

    new_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    binary_models.append(new_model)

    model_path = 'models/classification_wbc_binary_'+c+'_best'

    checkpoint = ModelCheckpoint(
        model_path+'.h5',
        monitor = 'val_accuracy',
        verbose = 1,
        mode = 'max', 
        save_best_only = True
    )

    EPOCHS = 100
    binary_models[-1].compile(loss = 'categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adam(), # 'rmsprop',
    metrics = ['accuracy'])

    model_history = binary_models[-1].fit(
        train_data,
        epochs = EPOCHS,
        validation_data = validation_data,
        verbose = 1,
        callbacks = [reduce_lr,lr_scheduler,checkpoint] # observe for the future that 'reduce_lr' and 'lr_scheduler' cannot be mixed
    )

    with open(model_path+'_history.pickle','wb') as h:
        pickle.dump(model_history.history,h,protocol=pickle.HIGHEST_PROTOCOL);

model_5a = tf.keras.models.load_model('models/classification_wbc_binary_EOSINOPHIL.h5')
model_5b = tf.keras.models.load_model('models/classification_wbc_binary_LYMPHOCYTE.h5')
model_5c = tf.keras.models.load_model('models/classification_wbc_binary_MONOCYTE.h5')
model_5d = tf.keras.models.load_model('models/classification_wbc_binary_NEUTROPHIL.h5')

preds_5a = model_5a.predict(test_data_unshuffled)[:,0][np.newaxis]
preds_5b = model_5b.predict(test_data_unshuffled)[:,0][np.newaxis]
preds_5c = model_5c.predict(test_data_unshuffled)[:,0][np.newaxis]
preds_5d = model_5d.predict(test_data_unshuffled)[:,0][np.newaxis]

preds_5 = np.concatenate([preds_5a, preds_5b, preds_5c, preds_5d],axis=0).T
preds_5 = [np.argmax(p) for p in preds_5]

accuracy_5 = np.mean([1 if preds_5[i]==test_data_unshuffled.classes[i] else 0 for i in range(len(preds_5))])
accuracy_5

print(make_confusion_matrix(test_data_unshuffled.classes, preds_5))

file_data = get_file_paths()

def flow_df(generator,file_data,shuffle=True):
    return generator.flow_from_dataframe(
        file_data,
        directory=None,
        x_col='file_path',
        y_col='classes',
        target_size=(240,240),
        classes=None,
        class_mode='binary',
        batch_size=60,
        shuffle=shuffle
    );

binary_models = []

for c in CATEGORIES[:1]:
    file_data['classes'] = file_data['category'].apply(lambda x: '0' if x==c else '1')
    file_data_train = file_data[file_data['split']=='train'].reset_index(drop=True)
    file_data_validation = file_data[file_data['split']=='validation'].reset_index(drop=True)
    file_data_test = file_data[file_data['split']=='test'].reset_index(drop=True)

    train_data = flow_df(training_datagen,file_data_train);
    validation_data = flow_df(test_val_datagen,file_data_validation);
    train_data_unshuffled = flow_df(training_datagen,file_data_train,shuffle=False);
    validation_data_unshuffled = flow_df(test_val_datagen,file_data_validation,shuffle=False);
    test_data_unshuffled = flow_df(test_val_datagen,file_data_test,shuffle=False);
    break
    
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(240, 240, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    binary_models.append(new_model)

    model_path = 'models/classification_wbc_binary2_'+c+'.h5'

    checkpoint = ModelCheckpoint(
        model_path,
        monitor = 'val_accuracy',
        verbose = 0,
        mode = 'max', 
        save_best_only = True
    )

    EPOCHS = 50
    lr_scheduler = LearningRateScheduler(lambda epoch: 5e-5 * 10**(2*epoch/EPOCHS))
    
    binary_models[-1].compile(loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(), # 'rmsprop',
    metrics = ['accuracy'])

    model_history = binary_models[-1].fit(
        train_data,
        epochs = EPOCHS,
        validation_data = validation_data,
        verbose = 1,
        callbacks = [lr_scheduler,checkpoint]
    )

    with open(model_path+'_history.pickle','wb') as h:
        pickle.dump(model_history.history,h,protocol=pickle.HIGHEST_PROTOCOL);

model_6a = tf.keras.models.load_model('models/classification_wbc_binary2_EOSINOPHIL.h5')
model_6b = tf.keras.models.load_model('models/classification_wbc_binary2_LYMPHOCYTE.h5')
model_6c = tf.keras.models.load_model('models/classification_wbc_binary2_MONOCYTE.h5')
model_6d = tf.keras.models.load_model('models/classification_wbc_binary2_NEUTROPHIL.h5')

preds_6a = model_6a.predict(test_data_unshuffled)
preds_6b = model_6b.predict(test_data_unshuffled)
preds_6c = model_6c.predict(test_data_unshuffled)
preds_6d = model_6d.predict(test_data_unshuffled)

preds_6 = np.concatenate([preds_6a, preds_6b, preds_6c, preds_6d],axis=1)
preds_6 = [np.argmin(p) for p in preds_6]

combined_accuracy_binary = np.mean([1 if preds_6[i]==test_data_unshuffled_2.classes[i] else 0 for i in range(len(preds_6))])
combined_accuracy_binary

print(make_confusion_matrix(test_data_unshuffled_2.classes, preds_6))

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
(IMG_HEIGHT,IMG_WIDTH) = (240,240)

train_data_7 = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH+'/train',
    labels = 'inferred',
    label_mode = 'int',
    class_names = None,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    shuffle=True,
    seed=84,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

validation_data_7 = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH+'/validation',
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    shuffle=True,
    seed=84,
)

test_data_unshuffled_7 = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH+'/test',
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    shuffle=False, # easier if we shuffle only when we're ready to avoid gotchas
    seed=84,
)

train_data_7 = train_data_7.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE) #.batch(batch_size)
validation_data_7 = validation_data_7.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
test_data_unshuffled_7 = test_data_unshuffled_7.cache().prefetch(buffer_size = AUTOTUNE)

base_model = tf.keras.applications.Xception(
    weights='imagenet',
    input_shape=(240, 240, 3),
    include_top=False)
base_model.trainable = False

inputs_new = tf.keras.Input(shape=(240, 240, 3))
x = tf.keras.applications.xception.preprocess_input(inputs_new) # gives us values in the range [-1,1]
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs_new = tf.keras.layers.Dense(4)(x) # ,activation='softmax'

model_7 = tf.keras.Model(inputs_new, outputs_new)

model_7.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)

model_7.fit(
    train_data_7,
    validation_data = validation_data_7,
    batch_size = 32,
    epochs = 10
)

EPOCHS = 10

base_model.trainable = True # unfreezing *all* the layers unless there are any BatchNorms in there

model_7.compile(
    optimizer = tf.keras.optimizers.Adam(1e-5),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 2,
    mode = 'min',
    restore_best_weights = True
)

lr_scheduler = LearningRateScheduler(
    lambda epoch: 5e-5 * 10**(1*epoch/EPOCHS)
)

model_7.fit(
    train_data_7,
    validation_data = validation_data_7,
    batch_size = 32,
    epochs = EPOCHS,
    callbacks = [early_stopping,lr_scheduler]
)

def learning_trajectory(epochs, start=1e-5, end=1e-3, mode='linear'):
    # modes can be linear, exponential, plateau
    if mode == 'linear':
        return LearningRateScheduler(lambda epoch: start + ((end-start)*epoch/epochs))
    elif (mode == 'exp') or (mode == 'exponential'):
        return LearningRateScheduler(lambda epoch: start * 10**(np.log10(end/start)*(epoch)/(epochs-1)))
    #elif mode == 'plateau':
    #    return LearningRateScheduler(lambda epoch: start * np.log10(epoch/epochs)/np.log10(end/start))

with tf.device('CPU'):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(),  # 'horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(-0.3, 0.3)
    ])

for l in range(25, 35):
    print(l, base_model.layers[l].output.shape, base_model.layers[l].name) #.summary() #.layers[30] #.output.shape[1]

base_model = tf.keras.applications.Xception(
    weights='imagenet',
    input_shape=(240, 240, 3),
    include_top=False
)

# here we can make the model return whichever layer we want
base_model_2 = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[30].output)
base_model_2.trainable = True

inputs_new = tf.keras.Input(shape=(240, 240, 3))
x = data_augmentation(inputs_new)  # data augmentation for tf.Data.datasets
x = tf.keras.applications.xception.preprocess_input(x)  # gives us values in the range [-1,1]
x = base_model_2(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs_new = tf.keras.layers.Dense(4)(x)  # activation='softmax'

model_8 = tf.keras.Model(inputs_new, outputs_new)

EPOCHS = 100

model_8.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=4,
    mode='min',
    restore_best_weights=True
)

lr_scheduler = learning_trajectory(EPOCHS, 1e-5, 1e-3, 'exp')

history = model_8.fit(
    train_data,
    validation_data=validation_data,
    batch_size=32,
    epochs=EPOCHS,
    callbacks=[early_stopping, lr_scheduler]
)

tf.keras.models.save_model(model_8, 'models/classification_xception_multiclass.h5')

preds_8 = [np.argmax(p) for p in model_8.predict(test_data_unshuffled_7, verbose=0)]
true_classes_8 = test_data_unshuffled_7.map(lambda x,y: y).unbatch().batch(600) 
true_classes_8 = iter(true_classes_8).next().numpy() # to prove they are indeed the same in the new dataset format

accuracy_8 = np.mean([1 if preds_8[i]==true_classes_8[i] else 0 for i in range(len(preds_8))])
accuracy_8

print(make_confusion_matrix(true_classes_8, preds_8))

from PIL import Image

def augmenting_model(input_shape = (240, 240)):
    input_shape = input_shape+(3,)
    with tf.device('CPU'):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(-0.3,0.3)
        ])
    inputs = tf.keras.Input(shape=input_shape)
    outputs = data_augmentation(inputs)
    return tf.keras.Model(inputs, outputs)

def augment_process(data, path, labels, repeat=1): # we perform augmentation as a separate exercise, to bypass GPU/CPU issues
    os.makedirs(path, exist_ok = True)
    for c in labels: os.makedirs(os.path.join(path, c), exist_ok=True)

    aug_model = augmenting_model()
    for i, batch in enumerate(data.repeat(repeat)):
        x, y = batch
        x = aug_model(x.numpy())
        for j in range(len(x)):
            l = y[j].numpy()
            if (CATEGORIES[l] in labels) or (l in labels):
                im = Image.fromarray(x[j].numpy().astype(np.uint8))
                im.save(os.path.join(path, CATEGORIES[l], ('0000'+str((i*len(x))+j))[-5:]+'.jpeg'))

    data_aug = tf.keras.utils.image_dataset_from_directory( # read back in as augmented TF dataset
        path,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=(240, 240),
        shuffle=True,
        seed=84,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    return data_aug

train_data_aug = augment_process(
    train_data_7,
    os.path.join(DATA_PATH,'augmented_train'),
    CATEGORIES,
    repeat=1
)

def augment_data(data, dataset_size=2500, batch_size=32, input_shape=(240, 240), write_path='augmented'):
    model_aug = augmenting_model(input_shape = input_shape)
    aug_data_x, aug_data_y = deque(), deque()
    for i, batch in enumerate(data):
        if (i > 0) and (i%50 == 0): print(i, 'augmented batches processed...')
        xt, yt = batch
        xt = np.round(model_aug(xt).numpy()*1.,0)
        yt = yt.numpy()*1.
        for j in range(len(xt)):
            aug_data_x.append(xt[j])
            aug_data_y.append(np.int32(yt[j]))
    aug_data_x, aug_data_y = np.asarray(aug_data_x), np.asarray(aug_data_y)
    train_data_augmented = []
    for i in range((len(aug_data_x)//dataset_size)+1):
        imin, imax = dataset_size*i, min(len(aug_data_x),dataset_size*(i+1))
        tf_data_x = tf.data.Dataset.from_tensor_slices(aug_data_x[imin:imax])
        tf_data_y = tf.data.Dataset.from_tensor_slices(aug_data_y[imin:imax])
        tf_data = tf.data.Dataset.zip((tf_data_x, tf_data_y)).cache().batch(batch_size).shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        train_data_aug.append(tf_data)
    return train_data_aug

train_data_aug = augment_data(train_data)

def make_binary_datasets(data, pos_label):
    return data.map(lambda x,y: (x,tf.map_fn(lambda z: 1 if z==pos_label else 0, y)))

train_data_binary, train_data_binary_aug, val_data_binary, test_data_binary = [],[],[],[]
for j in range(len(CATEGORIES)):
    train_data_binary.append(make_binary_datasets(train_data_7, j))
    train_data_binary_aug.append(make_binary_datasets(train_data_aug, j))
    val_data_binary.append(make_binary_datasets(validation_data_7, j))
    test_data_binary.append(make_binary_datasets(test_data_unshuffled_7, j))

EPOCHS = 20
lrs = (1e-4, 1e-2)

def make_binary_model(input_shape=(240,240), base_model_trainable=True, base_model_last_layer=None):
    input_shape = input_shape+(3,)

    base_model = tf.keras.applications.Xception(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False
    )

    # here we can make the model return whichever layer we want
    if base_model_last_layer is not None:
        base_model_test = tf.keras.Model(inputs = base_model.input, outputs = base_model.get_layer(base_model_last_layer).output)
    base_model_test.trainable = base_model_trainable

    inputs_new = tf.keras.Input(shape = input_shape)
    x = tf.keras.applications.xception.preprocess_input(inputs_new)  # gives us values in the range [-1,1]
    x = base_model_test(x, training = False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(16, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs_new = tf.keras.layers.Dense(1)(x)
    model_new = tf.keras.Model(inputs_new, outputs_new)

    model_new.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
        metrics = ['accuracy']
    )

    return model_new

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', restore_best_weights=True)
lr_scheduler = learning_trajectory(EPOCHS, lrs, 'exp')

models_9, history_9 = [],[]

for i in range(4):
    models_9.append(make_binary_model(base_model_last_layer=base_model.layers[30])
    history_9.append(models_binary[i].fit(
        train_data_binary_aug[i],  # augmented training data
        validation_data=val_data_binary[i],  # non-augmented validation data :)
        batch_size=32,
        epochs=EPOCHS,
        callbacks=[lr_scheduler, early_stopping]
    ))
    tf.keras.models.save_model(models_9[i], 'models/classification_xception_binary_'+str(i)+'.h5')

FINE_EPOCHS = 20
lrs = (1e-6, 1e-6)
lr_scheduler = learning_trajectory(FINE_EPOCHS, lrs[0], lrs[1], 'exp')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, mode='max', restore_best_weights=True)

history_binary_ft = []

for i in range(4):
    history_binary_ft.append(models_9[i].fit(
        train_data_binary_aug[i], # augmented training data
        validation_data=val_data_binary[i], # non-augmented validation data :)
        batch_size=32,
        epochs=FINE_EPOCHS,
        initial_epoch=history_binary[i].epoch[-1]+1,
        callbacks=[lr_scheduler, early_stopping]
    ))
    tf.keras.models.save_model(models_9[i], 'models/classification_xception_binary_fine_'+str(i)+'.h5')

preds_9a = models_9[0].predict(test_data_unshuffled_7, verbose=0)[:,0][np.newaxis]
preds_9b = models_9[1].predict(test_data_unshuffled_7, verbose=0)[:,0][np.newaxis]
preds_9c = models_9[2].predict(test_data_unshuffled_7, verbose=0)[:,0][np.newaxis]
preds_9d = models_9[3].predict(test_data_unshuffled_7, verbose=0)[:,0][np.newaxis]

preds_9 = np.concatenate([preds_9a, preds_9b, preds_9c, preds_9d], axis=0).T
preds_9 = [np.argmax(p) for p in preds_9]

accuracy_9 = np.mean([1 if preds_9[i]==true_classes_8[i] else 0 for i in range(len(preds_9))])
accuracy_9

print(make_confusion_matrix(true_classes_8, preds_9))

preds_9_8 = [2 if preds_8[i]==2 else preds_9[i] for i in range(len(preds_9))]
accuracy_9_8 = np.mean([1 if preds_9_8[i]==true_classes_8[i] else 0 for i in range(len(preds_9))])
accuracy_9_8

print(make_confusion_matrix(true_classes_8, preds_9_8))

tf.math.confusion_matrix(
    true_classes_8, preds_9_8,
    num_classes=4
)

import io
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

def make_triplet_loss_model(input_shape=(240,240), base_model_trainable=True, base_model_last_layer=None):
    input_shape = input_shape+(3,)

    base_model = tf.keras.applications.Xception(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False
    )

    # here we can make the model return whichever layer we want
    if base_model_last_layer is not None:
        base_model_test = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer(base_model_last_layer).output)
    base_model_test.trainable = base_model_trainable

    inputs_new = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.xception.preprocess_input(inputs_new) # gives us values in the range [-1,1]
    x = base_model_test(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x) # lets experiment with some stronger regularization here 
    outputs_new = tf.keras.layers.Dense(728)(x)
    model_new = tf.keras.Model(inputs_new, outputs_new)

    model_new.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tfa.losses.TripletSemiHardLoss()
    )

    return model_new

EPOCHS = 20
lrs = (1e-4, 1e-2)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True)
lr_scheduler = learning_trajectory(EPOCHS, lrs, 'exp')

model_tl = make_triplet_loss_model(base_model_last_layer='block13_sepconv2_act')
model_tl.fit(
    train_data_aug, # augmented training data
    validation_data=validation_data, # non-augmented validation data :)
    batch_size=32,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stopping]
)

tf.keras.models.save_model(model_tl, 'models/xception_tl') # TO DO - register the triplet loss function as a custom model object

EPOCHS = 50
lrs = (1e-5, 1e-2)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=25, mode='max', restore_best_weights=True)
lr_scheduler = learning_trajectory(EPOCHS, lrs[0], lrs[1], 'exp')

for i, train_batch in enumerate(train_data_aug):
    train_x, train_y = train_batch
    pred_tl = model_tl.predict(train_x, verbose=0)[0]
    train_y_real = train_y.numpy() if i == 0 else np.concatenate([train_y_real, train_y.numpy()], axis=0)
    pred_tl_train = pred_tl if i == 0 else np.concatenate([pred_tl_train, pred_tl], axis=0)

for i, val_batch in enumerate(validation_data_7):
    val_x, val_y = val_batch
    pred_tl = model_tl.predict(val_x, verbose = 0)[0]
    val_y_real = val_y.numpy() if i == 0 else np.concatenate([val_y_real, val_y.numpy()], axis=0)
    pred_tl_val = pred_tl if i == 0 else np.concatenate([pred_tl_val, pred_tl], axis=0)

for i, test_batch in enumerate(test_data_unshuffled_7):
    test_x, test_y = test_batch
    pred_tl = model_tl.predict(test_x, verbose = 0)[0]
    test_y_real = test_y.numpy() if i == 0 else np.concatenate([test_y_real, test_y.numpy()], axis=0)
    pred_tl_test = pred_tl if i == 0 else np.concatenate([pred_tl_test, pred_tl], axis=0)

embedding_tl_train = tf.data.Dataset.from_tensor_slices((pred_tl_train, train_y_real)).batch(32)
embedding_tl_val = tf.data.Dataset.from_tensor_slices((pred_tl_val, val_y_real)).batch(32)
embedding_tl_test = tf.data.Dataset.from_tensor_slices((pred_tl_test, test_y_real)).batch(32)

model_tl_classify = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# compile the model
model_tl_classify.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history_tl = model_tl_classify.fit(
    embedding_tl_train,
    epochs=100,
    validation_data=embedding_tl_val,
    callbacks=[early_stopping, lr_scheduler]
)

for i, test_batch in enumerate(embedding_tl_test):
    test_x, test_y = test_batch
    pred_tl = model_tl_classify.predict(test_x, verbose=0)
    test_y_real = test_y.numpy() if i == 0 else np.concatenate([test_y_real, test_y.numpy()], axis=0)
    pred_tl_all = pred_tl if i == 0 else np.concatenate([pred_tl_all, pred_tl], axis=0)

pred_tl_all = [np.argmax(p) for p in pred_tl_all]

print(pred_tl_all)  # delete this later

print(make_confusion_matrix(test_y_real, pred_tl_all))

tf.math.confusion_matrix(
    test_y_real,
    pred_tl_all,
    num_classes=4
)

print('Accuracy:', '{:.3%}'.format(np.mean([1 if diff == 0 else 0 for diff in (pred_tl_all - test_y_real)])))

zz = [2 if preds_8[i]==2 else pred_tl_all[i] for i in range(len(preds_8))]
print('Accuracy:', '{:.3%}'.format(np.mean([1 if diff == 0 else 0 for diff in (zz - test_y_real)])))

train_data_aug_v2 = augment_process(
    train_data_7,
    os.path.join(DATA_PATH, 'augmented_train_v2'),
    [CATEGORIES[j] for j in [2,3]],
    repeat=4
).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

validation_data = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH+'/validation',
    labels='inferred',
    label_mode='int',
    batch_size=None,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    shuffle=True,
    seed=84,
)

# subset validation data to keep only classes 2 and 3
x_tensors,y_tensors = [],[]
labels = [2,3]
for i,t in enumerate(validation_data):
    if t[1].numpy() in labels:
        x_tensors.append(t[0])
        y_tensors.append(t[1])

tf_labels = tf.constant(labels) 

validation_data_v2 = tf.data.Dataset.from_tensor_slices((x_tensors, y_tensors)).batch(32)
validation_data_v2 = validation_data_v2.map(lambda x, y: (x, tf.map_fn(lambda z: tf.cast(tf.where(tf.equal(z,tf_labels))[0][0], dtype='int32'), y)))
validation_data_v2 = validation_data_v2.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

test_data = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH+'/test',
    labels='inferred',
    label_mode='int',
    batch_size=None,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    shuffle=False,
    seed=84,
)

# subset validation data to keep only classes 2 and 3
x_tensors,y_tensors = [],[]
labels = [2,3]
for i,t in enumerate(test_data):
    if t[1].numpy() in labels:
        x_tensors.append(t[0])
        y_tensors.append(t[1])
    
@tf.function
def map_labels(labels,z):
    tf_labels = tf.constant(labels)
    return tf.cast(tf.where(tf.equal(z,tf_labels))[0][0],dtype='int32')

test_data = tf.data.Dataset.from_tensor_slices((x_tensors, y_tensors)).batch(32)
test_data = test_data.map(lambda x, y: (x, tf.map_fn(lambda z: map_labels(labels,z), y)))
test_data = test_data.cache().prefetch(buffer_size = AUTOTUNE)

EPOCHS = 20
lrs = (1e-7, 1e-5)  # using a VERY low learning rate here to address observed sensitivity of the loss function
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
lr_scheduler = learning_trajectory(EPOCHS, lrs[0], lrs[1], 'exp')

model_tl_binary = make_triplet_loss_model(base_model_last_layer='block13_sepconv2_act')
model_tl_binary.fit(
    train_data_aug_v2, # augmented training data
    validation_data=validation_data_v2, # non-augmented validation data :)
    batch_size=32,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, early_stopping]
)

tf.keras.models.save_model(model_tl_binary, 'models/xception_tl_binary')

model_tl_binary = tf.keras.models.load_model('models/xception_tl_binary')

for i, train_batch in enumerate(train_data_aug_v2):
    train_x, train_y = train_batch
    pred_tl = model_tl_binary.predict(train_x, verbose = 0)
    train_y_real_binary = train_y.numpy() if i == 0 else np.concatenate([train_y_real_binary, train_y.numpy()], axis = 0)
    pred_tl_train_binary = pred_tl if i == 0 else np.concatenate([pred_tl_train_binary, pred_tl], axis = 0)

for i, val_batch in enumerate(validation_data_v2):
    val_x, val_y = val_batch
    pred_tl = model_tl_binary.predict(val_x, verbose = 0)
    val_y_real_binary = val_y.numpy() if i == 0 else np.concatenate([val_y_real_binary, val_y.numpy()], axis = 0)
    pred_tl_val_binary = pred_tl if i == 0 else np.concatenate([pred_tl_val_binary, pred_tl], axis = 0)

for i, test_batch in enumerate(test_data):  # we actually test on all 4 classes again
    test_x, test_y = test_batch
    pred_tl = model_tl_binary.predict(test_x, verbose = 0)
    test_y_real_binary = test_y.numpy() if i == 0 else np.concatenate([test_y_real_binary, test_y.numpy()], axis = 0)
    pred_tl_test_binary = pred_tl if i == 0 else np.concatenate([pred_tl_test_binary, pred_tl], axis = 0)

embedding_tl_binary_train = tf.data.Dataset.from_tensor_slices((pred_tl_train_binary, train_y_real_binary)).batch(32)
embedding_tl_binary_val = tf.data.Dataset.from_tensor_slices((pred_tl_val_binary, val_y_real_binary)).batch(32)
embedding_tl_binary_test = tf.data.Dataset.from_tensor_slices((pred_tl_test_binary, test_y_real_binary)).batch(32)

model_tl_binary_classify = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the model
model_tl_binary_classify.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = ['accuracy']
)

EPOCHS = 50
lrs = (1e-6, 1e-4)

early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 10, mode = 'max', restore_best_weights = True)
lr_scheduler = learning_trajectory(EPOCHS, lrs[0], lrs[1], 'exp')

history_tl = model_tl_binary_classify.fit(
    embedding_tl_binary_train,
    epochs = EPOCHS,
    validation_data = embedding_tl_binary_val,
    callbacks = [early_stopping, lr_scheduler]
)

tf.keras.models.save_model(model_tl_binary_classify, 'models/xception_tl_binary_classify')

for i, test_batch in enumerate(embedding_tl_binary_test):
    test_x, test_y = test_batch
    pred_tl = model_tl_binary_classify.predict(test_x, verbose = 0)
    test_y_real_binary = test_y.numpy() if i == 0 else np.concatenate([test_y_real_binary, test_y.numpy()], axis = 0)
    pred_tl_all_binary = pred_tl if i == 0 else np.concatenate([pred_tl_all_binary, pred_tl], axis = 0)

pred_tl_all_binary = 2+np.asarray([int(p) for p in np.round(pred_tl_all_binary.flatten(),0)])
pred_tl_all_binary_final = np.concatenate([pred_tl_all[:300],pred_tl_all_binary],axis=0)
print('Accuracy:','{:.3%}'.format(np.mean([1 if diff == 0 else 0 for diff in (pred_tl_all_binary_final - test_y_real)])))

print(pred_tl_all_binary_final)  # can remove this later

print(make_confusion_matrix(test_y_real, pred_tl_all_binary_final))

tf.math.confusion_matrix(
    test_y_real_binary+2,
    pred_tl_all_binary,
    num_classes=4
)




cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/content/drive/MyDrive/hymenoptera_data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

visualize_model(model_ft)

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

from google.colab import drive
drive.mount('/content/drive')

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)

        image_paths = []
bee_train_dir = '/content/drive/MyDrive/hymenoptera_data/hymenoptera_data/train/bees'
ant_train_dir = '/content/drive/MyDrive/hymenoptera_data/hymenoptera_data/train/ants'

bee_val_dir = '/content/drive/MyDrive/hymenoptera_data/hymenoptera_data/val/bees'
ant_val_dir = '/content/drive/MyDrive/hymenoptera_data/hymenoptera_data/val/ants'

images_train = []
labels_train = []

images_val = []
labels_val = []

for filename in os.listdir(bee_train_dir):
  temp = os.path.join(bee_train_dir, filename)
  images_train.append(str(temp))
  labels_train.append(0)

for filename in os.listdir(ant_train_dir):
  temp = os.path.join(ant_train_dir, filename)
  images_train.append(str(temp))
  labels_train.append(1)

labels_train.remove(1)


for filename in os.listdir(bee_val_dir):
  temp = os.path.join(bee_val_dir, filename)
  images_val.append(str(temp))
  labels_val.append(0)

for filename in os.listdir(ant_val_dir):
  temp = os.path.join(ant_val_dir, filename)
  images_val.append(str(temp))
  labels_val.append(1)
  
  visualize_model_predictions(
    model_conv,
    img_path='/content/drive/MyDrive/hymenoptera_data/hymenoptera_data/train/ants/1030023514_aad5c608f9.jpg'
)

plt.ioff()
plt.show()

def resnet18_features(img_path):
    model = models.resnet18(weights=True)                  # Remove the final fully-connected layer to get the 512 dimensional features
    modules = list(model.children())[:-1]                  # The final layer converts the features to 1000 dimensions
    model = torch.nn.Sequential(*modules)                  # Makes a list of all the layers except the final layer then sequentially stack them
    model.eval()                                           # Set the model to evaluation mode
    img = Image.open(img_path).convert('RGB')                          # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    img = transform(img)

    img = img.unsqueeze(0)
    with torch.no_grad():
        features = model(img)           # Get the ResNet18 features for the image
    features = features.squeeze()
    return features.numpy()             # Remove the batch dimension and return the features

img_train_features = []
coutn = 0
for i in images_train:
  if coutn == 172:                      # Encountered some error with the dimension for the 52nd image from the ants training dataset, so decided to just discard it.
    coutn +=1
    continue
  else:
    dee = resnet18_features(i)
    coutn += 1
    # print(coutn)
    img_train_features.append(dee)

img_val_features = []
for i in images_val:
  dee = resnet18_features(i)
  img_val_features.append(dee)

img_train_features = np.array(img_train_features)
img_val_features = np.array(img_val_features)
print(img_train_features.shape, img_val_features.shape)

# Printing the shape of the training and validation sets to ensure they are correct

x_train = img_train_features
y_train = np.array(labels_train)
# print(x_train.shape, y_train.shape)

x_val = img_val_features
y_val = np.array(labels_val)
# print(x_val.shape, y_val.shape)

print(img_train_features.shape, img_val_features.shape)

param_grid_rbf = {'C' : np.logspace(-3,3,20),
                  'gamma': np.logspace(-3,3,20)}

scoring_f1 = 'f1'
scoring_accuracy = 'accuracy'

grid_search_rbf_f1 = GridSearchCV(estimator = SVC(kernel = 'rbf'), param_grid=param_grid_rbf, scoring=scoring_f1, cv=5)
grid_search_rbf_f1.fit(x_train, y_train)

grid_search_rbf_notf1 = GridSearchCV(estimator = SVC(kernel = 'rbf'), param_grid=param_grid_rbf, scoring=scoring_accuracy, cv=5)
grid_search_rbf_notf1.fit(x_train, y_train)

f1_best_score = grid_search_rbf_f1.best_score_
accuracy_best_score = grid_search_rbf_notf1.best_score_
print('best score for rbf kernel with accuracy scoring : ', accuracy_best_score)
print('best score for rbf kernel with f1 scoring : ', f1_best_score)

print('best parameters with f1 metric : ', grid_search_rbf_f1.best_params_)
best_rbf = grid_search_rbf_f1.best_params_
C = grid_search_rbf_f1.cv_results_['param_C'].reshape((len(param_grid_rbf['C']), len(param_grid_rbf['gamma'])))
gamma = grid_search_rbf_f1.cv_results_['param_gamma'].reshape((len(param_grid_rbf['C']), len(param_grid_rbf['gamma'])))
x, y = C, gamma
plt.contourf(x,y,grid_search_rbf_f1.cv_results_['mean_test_score'][::1].reshape((len(param_grid_rbf['C']), len(param_grid_rbf['gamma']))),levels = 25)
plt.scatter(best_rbf['C'], best_rbf['gamma'], s=15, c= 'r')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()

print('best parameters with accuracy metric : ', grid_search_rbf_notf1.best_params_)
best_rbf = grid_search_rbf_notf1.best_params_
C = grid_search_rbf_notf1.cv_results_['param_C'].reshape((len(param_grid_rbf['C']), len(param_grid_rbf['gamma'])))
gamma = grid_search_rbf_notf1.cv_results_['param_gamma'].reshape((len(param_grid_rbf['C']), len(param_grid_rbf['gamma'])))
x, y = C, gamma
plt.contourf(x,y,grid_search_rbf_notf1.cv_results_['mean_test_score'][::1].reshape((len(param_grid_rbf['C']), len(param_grid_rbf['gamma']))),levels = 25)
plt.scatter(best_rbf['C'], best_rbf['gamma'], s=15, c= 'r')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()

"""**Best Parameters for the rbf kernel are :**</br>
1) For accuracy metric: C = 0.1623776739188721, gamma = 0.004281332398719396 </br>
2) For f1 metric: C = 0.1623776739188721, gamma = 0.004281332398719396
"""

rbf_kernel_f1 = SVC(kernel = 'rbf', C = 0.1623776739188721, gamma = 0.004281332398719396)
rbf_kernel_f1.fit(x_train, y_train)

y_pred_rbf_f1 = rbf_kernel_f1.predict(x_val)
f1_score_rbf_f1 = f1_score(y_pred_rbf_f1, y_val)
print('f1 score on testing for rbf kernel : ',f1_score_rbf_f1)


rbf_kernel_notf1 = SVC(kernel = 'rbf', C = 0.1623776739188721, gamma = 0.004281332398719396 )
rbf_kernel_notf1.fit(x_train, y_train)

y_pred_rbf_notf1 = rbf_kernel_notf1.predict(x_val)
f1_score_rbf_notf1 = accuracy_score(y_pred_rbf_notf1, y_val)
print('accuracy score on testing for rbf kernel : ',f1_score_rbf_notf1)

param_grid_rf = {
    'max_depth': range(1,20,2),
    'max_features' : range(1,20,4)
}

scoring_f1 = 'f1'
scoring_accuracy = 'accuracy'

grid_search_rf_f1 = GridSearchCV(estimator = RandomForestClassifier(), param_grid=param_grid_rf, scoring=scoring_f1, cv=5)
grid_search_rf_f1.fit(x_train, y_train)

grid_search_rf_notf1 = GridSearchCV(estimator = RandomForestClassifier(), param_grid=param_grid_rf, scoring=scoring_accuracy, cv=5)
grid_search_rf_notf1.fit(x_train, y_train)

f1_best_score = grid_search_rf_f1.best_score_
accuracy_best_score = grid_search_rf_notf1.best_score_
print('best score for random forest with accuracy scoring : ', accuracy_best_score)
print('best score for random forest with f1 scoring : ', f1_best_score)

best_rf_f1 = grid_search_rf_f1.best_params_
print('best parameters with f1 metric : ', grid_search_rf_f1.best_params_)

max_depth = grid_search_rf_f1.cv_results_['param_max_depth'].reshape((len(param_grid_rf['max_depth']), len(param_grid_rf['max_features'])))
max_features = grid_search_rf_f1.cv_results_['param_max_features'].reshape((len(param_grid_rf['max_depth']), len(param_grid_rf['max_features'])))
x, y = max_depth, max_features
plt.contourf(x,y,grid_search_rf_f1.cv_results_['mean_test_score'][::1].reshape((len(param_grid_rf['max_depth']), len(param_grid_rf['max_features']))),levels = 25)
plt.scatter([best_rf_f1['max_depth']], [best_rf_f1['max_features']], s=15, c= 'r')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('max_features')
plt.ylabel('max_depth')
plt.colorbar()

best_rf_notf1 = grid_search_rf_notf1.best_params_
print('best parameters with f1 metric : ', grid_search_rf_notf1.best_params_)

max_depth = grid_search_rf_notf1.cv_results_['param_max_depth'].reshape((len(param_grid_rf['max_depth']), len(param_grid_rf['max_features'])))
max_features = grid_search_rf_notf1.cv_results_['param_max_features'].reshape((len(param_grid_rf['max_depth']), len(param_grid_rf['max_features'])))
x, y = max_depth, max_features
plt.contourf(x,y,grid_search_rf_notf1.cv_results_['mean_test_score'][::1].reshape((len(param_grid_rf['max_depth']), len(param_grid_rf['max_features']))),levels = 25)
plt.scatter([best_rf_notf1['max_depth']], [best_rf_notf1['max_features']], s=15, c= 'r')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('max_features')
plt.ylabel('max_depth')
plt.colorbar()

"""**Best Parameters for the Random Forest Classifier are :**</br>
1) For accuracy metric: max_depth = 5, max_features = 9 </br>
2) For f1 metric: max_depth = 19, max_features = 5
"""

rff_f1 = RandomForestClassifier(max_depth = 5, max_features = 9)
rff_f1.fit(x_train, y_train)

y_pred_f1 = rff_f1.predict(x_val)
f1_score_rf = f1_score(y_pred_f1, y_val)
print('f1 score for random forest on testing : ', f1_score_rf)

rff_notf1 = RandomForestClassifier(max_depth = 19, max_features = 5)
rff_notf1.fit(x_train, y_train)

y_pred_notf1 = rff_notf1.predict(x_val)
accuracy_score_rf = accuracy_score(y_pred_notf1, y_val)
print('accuracy score for random forest on testing : ',accuracy_score_rf)

"""# Observations and references:
**Resnet is a very useful model for feature extraction from images. After the feature extraction, the models each present us with the scores for the test set. The Accuracy and F1 scores for each of them are insignificantly nearby, but to be precise, the logistic regression model has the best scores for testing.** </br> </br>


2) RBF Kernel SVM: </br>
-----testing accuracy score is :  0.9630952380952381</br>
-----testing f1 score is :  0.963504766455077

3) Random Forest Classifier: </br>
-----testing accuracy score is :  0.954248366013072</br>
-----testing f1 score :  0.9640287769784173</br>

## Fine-tuning the convnet

Load a pretrained model and reset final fully connected layer.
"""

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 2)

# model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# # Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

### Train and evaluate
#It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.

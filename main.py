import streamlit as st
import torch
import torch.nn as nn
from torch.functional import F
from matplotlib import image as img
import numpy as np


# selecting device
device = "cuda" if torch.cuda.is_available() else "cpu"


# dictionaries form models
dict_pneumonia = {
    0: "Zdrowe płuca",
    1: "Zapalenie płuc"
}

dict_oct = {
    0: "Neowaskularuzacja naczyniówkowa",
    1: "Cukrzycowy obrzęk plamki",
    2: "Druzy tarczy nerwu wzrokowego",
    3: "Zdrowe oko"
}

dict_retina = {
    0: "0 poziom retinopatii cukrzycowej",
    1: "1 poziom retinopatii cukrzycowej",
    2: "2 poziom retinopatii cukrzycowej",
    3: "3 poziom retinopatii cukrzycowej",
    4: "4 poziom retinopatii cukrzycowej"
}


def predict(model, image):
    with torch.no_grad():
        image = torch.Tensor(image).reshape(1, 1, 28, -1)
        net_out = model(image)  # returns a list,
        predicted_class = torch.argmax(net_out)
        print('net_out: ', net_out)
    return predicted_class.item(), np.array(net_out)


def predictRGB(model, image):
    with torch.no_grad():
        image = torch.Tensor(image).reshape(1, 3, 28, -1)
        net_out = model(image)  # returns a list,
        predicted_class = torch.argmax(net_out)
        print('net_out: ', net_out)
    return predicted_class.item(), np.array(net_out)


def model_upload(network, model_path):
        model = network()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model


# model class for pneumonia prediction
class PneumoniaNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(20, 40, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(40)

        self.conv3 = nn.Conv2d(40, 80, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(80)

        self.conv4 = nn.Conv2d(80, 160, 3, 1)
        self.conv4_bn = nn.BatchNorm2d(160)

        self.fc1 = nn.Linear(7*7*160, 1000)  # ((((28-2)/2)-2)/2) = 5,..
        self.fc1_bn = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 500)
        self.fc2_bn = nn.BatchNorm1d(500)

        self.fc3 = nn.Linear(500, 50)
        self.fc3_bn = nn.BatchNorm1d(50)

        self.fc4 = nn.Linear(50, 2)

    def forward(self, X):
        X = self.conv1(X)
        X = F.relu(self.conv1_bn(X))  # 1

        X = F.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = F.relu(self.conv2_bn(X))

        X = self.conv3(X)
        X = F.relu(self.conv3_bn(X))  # 3

        X = self.conv4(X)
        X = F.relu(self.conv4_bn(X))  # 4

        X = X.view(-1, 7*7*160)

        X = self.fc1(X)
        X = F.relu(self.fc1_bn(X))
        X = self.fc2(X)
        X = F.relu(self.fc2_bn(X))
        X = self.fc3(X)
        X = F.relu(self.fc3_bn(X))
        X = self.fc4(X)
        return F.softmax(X, dim=1)


class OCTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(20, 40, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(40)

        self.conv3 = nn.Conv2d(40, 80, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(80)

        self.conv4 = nn.Conv2d(80, 160, 3, 1)
        self.conv4_bn = nn.BatchNorm2d(160)

        self.fc1 = nn.Linear(7*7*160, 1000)  # ((((28-2)/2)-2)/2) = 5,..
        self.fc1_bn = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 500)
        self.fc2_bn = nn.BatchNorm1d(500)

        self.fc3 = nn.Linear(500, 50)
        self.fc3_bn = nn.BatchNorm1d(50)

        self.fc4 = nn.Linear(50, 4)  # change

    def forward(self, X):
        X = self.conv1(X)
        X = F.relu(self.conv1_bn(X))  #1

        X = F.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = F.relu(self.conv2_bn(X))

        X = self.conv3(X)
        X = F.relu(self.conv3_bn(X))  #3

        X = self.conv4(X)
        X = F.relu(self.conv4_bn(X))  #4

        X = X.view(-1, 7*7*160)

        X = self.fc1(X)
        X = F.relu(self.fc1_bn(X))
        X = self.fc2(X)
        X = F.relu(self.fc2_bn(X))
        X = self.fc3(X)
        X = F.relu(self.fc3_bn(X))
        X = self.fc4(X)
        return F.softmax(X, dim=1)


class RetinaNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 63, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(63)

        self.conv4 = nn.Conv2d(63, 128, 3, 1)
        self.conv4_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(7*7*128, 1000)  # ((((28-2)/2)-2)/2) = 5,..
        self.fc1_bn = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 200)
        self.fc2_bn = nn.BatchNorm1d(200)

        self.fc3 = nn.Linear(200, 50)
        self.fc3_bn=nn.BatchNorm1d(50)

        self.fc4 = nn.Linear(50, 5)

    def forward(self, X):
        X = self.conv1(X)
        X = F.relu(self.conv1_bn(X))

        X = F.max_pool2d(X, 2, 2)

        X = self.conv2(X)
        X = F.relu(self.conv2_bn(X))

        X = self.conv3(X)
        X = F.relu(self.conv3_bn(X))

        X = self.conv4(X)
        X = F.relu(self.conv4_bn(X))

        X = X.view(-1, 7*7*128)

        X = self.fc1(X)
        X = F.relu(self.fc1_bn(X))
        X = self.fc2(X)
        X = F.relu(self.fc2_bn(X))
        X = self.fc3(X)
        X = F.relu(self.fc3_bn(X))
        X = self.fc4(X)
        return F.softmax(X, dim=1)


st.markdown("<h1 style='text-align: center; color: silver;'>System klasyfikacji obrazowań medycznych</h1>", unsafe_allow_html=True)

st.text("\n\n")

st.text("W celu skorzystania z systemu wybierz odpowiednią kategorię obrazów medycznych.\n"
        "Następnie załaduj zdjęcie diagnostyczne.\n\nPoniżej przykłady obrazów każdej kategorii:\n\n")


im_1 = img.imread('files/pneumonia_2.png')
im_2 = img.imread('files/oct_2.png')
im_3 = img.imread('files/retina_2.png')
st.sidebar.text("Sprawdź statystyki")
model = st.sidebar.selectbox("Wybierz model:", ['pneumoniamnist', 'octmnist', 'retinamnist'])

if model == 'pneumoniamnist':
    st.sidebar.write("v`grwh`")
st.image([im_1, im_2, im_3], clamp=True, width=230)




# for models
img_pneumonia_0 = np.load('files/img_pneumonia_0_p.npy')
img_pneumonia_1 = np.load('files/img_pneumonia_1_p.npy')

img_oct_0 = np.load('files/img_oct_0_p.npy')
img_oct_1 = np.load('files/img_oct_1_p.npy')
img_oct_2 = np.load('files/img_oct_2_p.npy')
img_oct_3 = np.load('files/img_oct_3_p.npy')

img_retina_0 = np.load('files/img_retina_0_p.npy')
img_retina_1 = np.load('files/img_retina_1_p.npy')
img_retina_2 = np.load('files/img_retina_2_p.npy')
img_retina_3 = np.load('files/img_retina_3_p.npy')
img_retina_4 = np.load('files/img_retina_4_p.npy')

# for pictures
img_pneumonia_0_ = np.load('files/img_pneumonia_0.npy')
img_pneumonia_1_ = np.load('files/img_pneumonia_1.npy')

img_oct_0_ = np.load('files/img_oct_0.npy')
img_oct_1_ = np.load('files/img_oct_1.npy')
img_oct_2_ = np.load('files/img_oct_2.npy')
img_oct_3_ = np.load('files/img_oct_3.npy')


model_pneumonia = model_upload(PneumoniaNetwork, 'files/model_pneumonia_91_the_best.pth')
model_oct = model_upload(OCTNetwork, 'files/model_oct_77.pth')
model_retina = model_upload(RetinaNetwork, 'files/model_retina_53_53.pth')

tab1, tab2, tab3 = st.tabs(["Płuca", "Siatkówka", "Dno oka"])

###################
# Tab for Pneumonia
###################
tab1.subheader("Próbki obrazów płuc:")

colp1, colp2, colp3 = tab1.columns(3)
with colp1:
    st.image(img_pneumonia_0_.reshape(28,28), width=100, clamp=True)
with colp2:
    pred_button = st.button("Klasyfikuj obraz")
with colp3:
    if pred_button:
        pred, out = predict(model_pneumonia, img_pneumonia_0)

        st.write(f'Klasyfikacja: {dict_pneumonia[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

tab1.markdown("""---""")

colp11, colp22, colp33 = tab1.columns(3)
with colp11:
    st.image(img_pneumonia_1_.reshape(28,28), width=100, clamp=True)
with colp22:
    pred_button = st.button("Klasyfikuj obraz", key=1)
with colp33:
    if pred_button:
        pred, out = predict(model_pneumonia, img_pneumonia_1)

        st.write(f'Klasyfikacja: {dict_pneumonia[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')


###################
# Tab for OCT
###################
tab2.subheader("Próbki obrazów siatkówki:")

colo1, colo2, colo3 = tab2.columns(3)
with colo1:
    st.image(img_oct_0_.reshape(28,28), width=100, clamp=True)
with colo2:
    pred_button = st.button("Klasyfikuj obraz", key=2)
with colo3:
    if pred_button:
        pred, out = predict(model_oct, img_oct_0)

        st.write(f'Klasyfikacja: {dict_oct[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

tab2.markdown("""---""")

colo11, colo22, colo33 = tab2.columns(3)
with colo11:
    st.image(img_oct_1_.reshape(28,28), width=100, clamp=True)
with colo22:
    pred_button = st.button("Klasyfikuj obraz", key=3)
with colo33:
    if pred_button:
        pred, out = predict(model_oct, img_oct_1)

        st.write(f'Klasyfikacja: {dict_oct[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

tab2.markdown("""---""")

colo111, colo222, colo333 = tab2.columns(3)
with colo111:
    st.image(img_oct_2_.reshape(28,28), width=100, clamp=True)
with colo222:
    pred_button = st.button("Klasyfikuj obraz", key=4)
with colo333:
    if pred_button:
        pred, out = predict(model_oct, img_oct_2)

        st.write(f'Klasyfikacja: {dict_oct[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

tab2.markdown("""---""")

colo1111, colo2222, colo3333 = tab2.columns(3)
with colo1111:
    st.image(img_oct_3_.reshape(28,28), width=100, clamp=True)
with colo2222:
    pred_button = st.button("Klasyfikuj obraz", key=5)
with colo3333:
    if pred_button:
        pred, out = predict(model_oct, img_oct_3)

        st.write(f'Klasyfikacja: {dict_oct[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

###################
# Tab for Retina
###################
tab3.subheader("Próbki obrazów dna oka:")
# print(np.reshape(img_retina_0,3,28,28))
colr1, colr2, colr3 = tab3.columns(3)
with colr1:
    st.image('files/ret_0.png', width=100, clamp=True)
with colr2:
    pred_button = st.button("Klasyfikuj obraz", key=6)
with colr3:
    if pred_button:
        pred, out = predictRGB(model_retina, img_retina_0)

        st.write(f'Klasyfikacja: {dict_retina[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

tab3.markdown("""---""")

colr11, colr22, colr33 = tab3.columns(3)
with colr11:
    st.image('files/ret_1.png', width=100, clamp=True)
with colr22:
    pred_button = st.button("Klasyfikuj obraz", key=7)
with colr33:
    if pred_button:
        pred, out = predictRGB(model_retina, img_retina_1)

        st.write(f'Klasyfikacja: {dict_retina[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

tab3.markdown("""---""")

colr111, colr222, colr333 = tab3.columns(3)
with colr111:
    st.image('files/ret_2.png', width=100, clamp=True)
with colr222:
    pred_button = st.button("Klasyfikuj obraz", key=8)
with colr333:
    if pred_button:
        pred, out = predictRGB(model_retina, img_retina_2)

        st.write(f'Klasyfikacja: {dict_retina[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

tab3.markdown("""---""")

colr1111, colr2222, colr3333 = tab3.columns(3)
with colr1111:
    st.image('files/ret_3.png', width=100, clamp=True)
with colr2222:
    pred_button = st.button("Klasyfikuj obraz", key=9)
with colr3333:
    if pred_button:
        pred, out = predictRGB(model_retina, img_retina_3)

        st.write(f'Klasyfikacja: {dict_retina[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

tab3.markdown("""---""")

colr11111, colr22222, colr33333 = tab3.columns(3)
with colr11111:
    st.image('files/ret_4.png', width=100, clamp=True)
with colr22222:
    pred_button = st.button("Klasyfikuj obraz", key=10)
with colr33333:
    if pred_button:
        pred, out = predictRGB(model_retina, img_retina_4)

        st.write(f'Klasyfikacja: {dict_retina[pred]}')
        st.write(f'Pewność: {round((np.max(out) * 100), 2)}%')

# Thanks to PP


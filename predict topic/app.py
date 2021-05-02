import numpy as np
from flask import Flask, request, jsonify, render_template
#from keras.models import load_model
import pickle

app = Flask(__name__)

model = pickle.load(open('modelnb.pkl','rb'))
#model2 = pickle.load(open('modellr.pkl','rb'))
#with open('tokenizer.pickle', 'wb') as handle:
#    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#model3 =load_model('model.h5')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def convert_unicode(txt):
    txt = unicodedata.normalize('NFC', txt)
    return txt

def text_preprocess(txt):
    # chuẩn hóa unicode
    txt = convert_unicode(txt)
    # chuẩn hóa cách gõ dấu tiếng Việt (òa -> oà, úy -> uý,...) và tách từ
    txt = rdrsegmenter.tokenize(txt)
    txt = ' '.join(' '.join(word for word in sentence) for sentence in txt)
    # đưa về lower
    txt = txt.lower()
    # xóa các ký tự không cần thiết
    txt = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',txt)
    # xóa khoảng trắng thừa
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def predict():
    title = str(request.form.values)

    #Trả về kết quả dự đoán với MultinomialNB
    prediction = model.predict([title])

    if prediction==0: prediction = 'Technology'
    if prediction==1: prediction = 'Travel'
    if prediction==2: prediction = 'Education'
    if prediction==3: prediction = 'Entertainment'
    if prediction==4: prediction = 'Science'
    if prediction==5: prediction = 'Business'
    if prediction==6: prediction = 'Law'
    if prediction==7: prediction = 'Health'
    if prediction==8: prediction = 'World'
    if prediction==9: prediction = 'Sport'
    if prediction==10: prediction = 'News'
    if prediction==11: prediction = 'Vehicle'
    if prediction==12: prediction = 'Life'
     
    output = prediction

    #Trả về kết quả dự đoán với Logistic Regression
    #prediction2 = model2.predict([title])

    #if prediction2==0: prediction = 'Công nghệ'
    #if prediction2==1: prediction = 'Du lịch'
    #if prediction2==2: prediction = 'Giáo dục'
    #if prediction2==3: prediction = 'Giải trí'
    #if prediction2==4: prediction = 'Khoa học'
    #if prediction2==5: prediction = 'Kinh doanh'
    #if prediction2==6: prediction = 'Pháp luật'
    #if prediction2==7: prediction = 'Sức khỏe'
    #if prediction2==8: prediction = 'Thế giới'
    #if prediction2==9: prediction = 'Thể thao'
    #if prediction2==10: prediction = 'Thời sự'
    #if prediction2==11: prediction = 'Xe'
    #if prediction2==12: prediction = 'Đời sống'
     
    #output2 = prediction2

    #Trả về kết quả dự đoán với Text-CNN
    #seq= loaded_tokenizer.texts_to_sequences([title])
    #padded = sequence.pad_sequences(seq, maxlen=16)
    #prediction3 = model3.predict(padded).argmax(axis=-1)

    #if prediction3==0: prediction = 'Công nghệ'
    #if prediction3==1: prediction = 'Du lịch'
    #if prediction3==2: prediction = 'Giáo dục'
    #if prediction3==3: prediction = 'Giải trí'
    #if prediction3==4: prediction = 'Khoa học'
    #if prediction3==5: prediction = 'Kinh doanh'
    #if prediction3==6: prediction = 'Pháp luật'
    #if prediction3==7: prediction = 'Sức khỏe'
    #if prediction3==8: prediction = 'Thế giới'
    #if prediction3==9: prediction = 'Thể thao'
    #if prediction3==10: prediction = 'Thời sự'
    #if prediction3==11: prediction = 'Xe'
    #if prediction3==12: prediction = 'Đời sống'
     
    #output3 = prediction3

    #Trả về kết quả dự đoán với PhoBERT
    #seq= loaded_tokenizer.texts_to_sequences([title])
    #padded = sequence.pad_sequences(seq, maxlen=16)
    #prediction4 = model4.predict(padded).argmax(axis=-1)

    #if prediction4==0: prediction = 'Công nghệ'
    #if prediction4==1: prediction = 'Du lịch'
    #if prediction4==2: prediction = 'Giáo dục'
    #if prediction4==3: prediction = 'Giải trí'
    #if prediction4==4: prediction = 'Khoa học'
    #if prediction4==5: prediction = 'Kinh doanh'
    #if prediction4==6: prediction = 'Pháp luật'
    #if prediction4==7: prediction = 'Sức khỏe'
    #if prediction4==8: prediction = 'Thế giới'
    #if prediction4==9: prediction = 'Thể thao'
    #if prediction4==10: prediction = 'Thời sự'
    #if prediction4==11: prediction = 'Xe'
    #if prediction4==12: prediction = 'Đời sống'
     
    #output4 = prediction4
    
    return render_template('index.html', prediction_text1='{}'.format(output)) #prediction_text2='Logistic Regression: {}'.format(output2), 
    #prediction_text3='Text-CNN: {}'.format(output3), prediction_text4='PhoBERT: {}'.format(output4)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([data.values()])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
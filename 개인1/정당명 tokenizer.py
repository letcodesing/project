from keras.preprocessing.text import Tokenizer
# political_parties_origin = ['더불어민주당', '미래통합당', '미래한국당', '더불어시민당', '정의당',	'국민의당', '열린민주당'	'국민의당'	'정의당' '새누리당 민주통합당	진보정의당	통합진보당	선진통일당 한나라당	통합민주당	자유선진당	친박연대	민주노동당	창조한국당']
political_parties_origin = ['더불어민주당 미래통합당 미래한국당 더불어시민당 정의당	국민의당 열린민주당	국민의당 정의당 새누리당 민주통합당	진보정의당 통합진보당 선진통일당 한나라당 통합민주당 자유선진당 친박연대 민주노동당	창조한국당']

token = Tokenizer()
token.fit_on_texts(political_parties_origin)
print(token.word_index)



political_parties_origin = token



political_parties_result= ['더불어민주당', '국민의힘','정의당','무소속']

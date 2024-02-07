import streamlit as st
import openai

# 英語名を日本語名に翻訳する関数
def translate_name(name):
    translation_dict = {
        "bob": "菩部",
        "alice": "亜李洲",
        "mike": "真偉区",
        "george": "丈二",
        "catherine": "伽佐凛",
        "john": "約翰",
        "mary": "瑪莉",
        "david": "大衛",
        "emily": "艾米莉",
        "william": "威廉",
        "sarah": "莎拉",
        "james": "詹姆士",
        "jennifer": "珍妮弗",
        "michael": "邁克爾",
        "linda": "琳達",
        "chris": "克里斯",
        "jessica": "潔西卡",
        "matthew": "馬修",
        "patricia": "帕特裏夏",
        "daniel": "丹尼爾",
        "elizabeth": "伊莉莎白",
        "ryan": "萊恩",
        "susan": "蘇珊",
        "justin": "賈斯汀",
        "karen": "凱倫",
        "andrew": "安德魯",
        "lauren": "勞倫",
        "thomas": "湯瑪斯",
        "angela": "安琪拉",
        "pierre": "皮埃爾",
        "marie": "瑪麗",
        "jean": "讓",
        "sophie": "蘇菲",
        "jacques": "雅克",
        "amelie": "艾美莉",
        "antoine": "安托萬",
        "claire": "克萊爾",
        "louis": "路易",
        "julie": "朱莉",
        "alexandre": "亞歷山大",
        "camille": "卡米",
        "olivier": "奧利維耶",
        "margot": "瑪格特",
        "luc": "盧克",
        # ドイツ人の名前
        "johann": "約翰",
        "maria": "瑪麗亜",
        "paul": "保羅",
        "sophia": "索菲亜",
        "franz": "弗蘭茨",
        "anna": "安娜",
        "karl": "卡爾",
        "laura": "羅拉",
        "heinrich": "海因裏希",
        "elena": "艾倫娜",
        "max": "馬克斯",
        "sandra": "珊卓拉",
        "peter": "彼得",
        "emma": "艾瑪",
        "thomas": "托馬斯",
        "julia": "尤莉亞",
        "wilhelm": "威廉",
        "hannah": "漢娜",
        "simon": "西蒙",
        "lisa": "麗莎",
        # 韓国人の名前
        "seung": "昇",
        "hyun": "賢",
        "min": "珉",
        "ji": "智",
        "hwan": "煥",
        "soo": "秀",
        "jin": "進",
        "hee": "熙",
        "kyung": "慶",
        "joon": "俊",
        "young": "榮",
        "jiwoo": "智宇",
        "jung": "貞",
        "sun": "善",
        "tae": "泰",
        "jinwoo": "進宇",
        "eun": "恩",
        "sang": "尚",
        "jae": "在",
        "mi": "美",
        # インド人の名前
        "rahul": "拉布尔",
        "priya": "普丽娅",
        "anil": "阿尼尔",
        "divya": "蒂芙娅",
        "vikram": "维克拉姆",
        "nisha": "妮莎",
        "suresh": "苏雷什",
        "deepak": "迪帕克",
        "neha": "涅哈",
        "raj": "拉吉",
        "anita": "阿妮塔",
        "akash": "阿卡什",
        "poonam": "普纳姆",
        "arjun": "阿尔金",
        "kavita": "卡维塔",
        "vijay": "维杰",
        "swati": "斯瓦蒂",
        "rohit": "罗希特",
        "preeti": "普莉蒂",
        "amit": "阿米特",
        # ロシア人の名前
        "ivan": "伊凡",
        "olga": "奥尔加",
        "vladimir": "弗拉基米尔",
        "natalia": "纳塔莉娅",
        "dmitry": "德米特里",
        "svetlana": "斯韦特拉娜",
        "alexander": "亚历山大",
        "maria": "玛丽亚",
        "sergei": "谢尔盖",
        "anna": "安娜",
        "yuri": "尤里",
        "tatiana": "塔季扬娜",
        "maxim": "马克西姆",
        "irina": "伊琳娜",
        "oleg": "奥列格",
        "elena": "叶琳娜",
        "igor": "伊戈尔",
        "anastasia": "阿纳斯塔西娅",
        "alexey": "亚历克谢",
        "yulia": "尤利娅",
        "nikolai": "尼古拉",
        "marina": "玛丽娜",
        "andrei": "安德烈",
        "evgenia": "叶甫根尼娅",
        # 他の名前に対する当て字を追加
    }
    # データベースから当て字を取得する。見つからない場合はそのまま返す。
    japanese_name = translation_dict.get(name.lower(), name)
    return japanese_name

# ChatGPTを使用して名前を生成する関数
def generate_name(prompt):
    openai.api_key = "your-api-key"  # OpenAI APIキーを設定
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=50,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response.choices[0].text.strip()

def main():
    st.title("Name Translator & Generator")

    # 英語の名前の入力
    english_name = st.text_input("Enter an English name:")

    if st.button("Translate"):
        # 名前を日本語に翻訳
        japanese_name = translate_name(english_name)
        st.write("Japanese name:", japanese_name)

    if st.button("Generate"):
        # 名前を生成
        prompt = "Generate a name for a character: " + english_name
        generated_name = generate_name(prompt)
        st.write("Generated name:", generated_name)

if __name__ == "__main__":
    main()

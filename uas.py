import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
sns.set(style="whitegrid")

st.markdown(
'''
<style>
    .stApp {
   background-color: white;
    }
 
       .stWrite,.stMarkdown,.stTextInput,h1, h2, h3, h4, h5, h6 {
            color: purple !important;
        }
</style>
''',
unsafe_allow_html=True
)

st.title("Analisis Conjoint of Data Pizza")
st.markdown('dibuat oleh: Joseph F.H. (20234920002)')
st.write('''
conjoint analysis merupakan merode decomposional yang mengestimasikan preference dari suatu consumer berdasarkan level dari alternatif suatu atribut (Vithala R. Rao, 2014)
''')


st.write('''
data yang digunakan merupakan data ranking pizza yang diambil dari kaggle  \n
berikut link dari data: https://www.kaggle.com/datasets/sachinsin8h/pizza-attributes-dataset-for-conjoint-analysis/data  \n
berikut preview dari data  \n
         dat=pd.read_csv('pizza_data.csv')
    dat=pd.read_csv('pizza_data.csv')
    df_attributes = dat.copy()
    df_attributes = df_attributes.drop(columns=['ranking'], errors='ignore')
    df_attributes['price']  = df_attributes['price'].str.replace('$','').astype(float)

    levels_dict = {col: sorted(df_attributes[col].dropna().unique().tolist()) 
               for col in df_attributes.columns}

    st.write("---Attribute Levels---")
    for a, lv in levels_dict.items():
        st.write(f"{a:10}: {lv}  ({len(lv)} levels)")
''')

dat=pd.read_csv('pizza_data.csv')


dat

df_attributes = dat.copy()
df_attributes = df_attributes.drop(columns=['ranking'], errors='ignore')
df_attributes['price']  = df_attributes['price'].str.replace('$','').astype(float)

levels_dict = {col: sorted(df_attributes[col].dropna().unique().tolist()) 
               for col in df_attributes.columns}

st.write("\n---Attribute Levels---")
for a, lv in levels_dict.items():
    st.write(f"{a:10}: {lv}  ({len(lv)} levels)")
st.write('''
data cocok untuk conjoint analysis karena terdapat beberapa variabel yang dapat mempengaruhi seberapa suka orang terhadap makanan dan juga terdapat variabel rank yang menunjukan prefrence   \n
data memiliki 9 atribut, dan atribut brand, price, dan weight memiliki empat level, dan atribut crust, cheese, size, toppings, spicy memiliki dua atribut. 
rating disaji dengan variuabel rank dimana disaji dimanana semakin kecil rank semakin disukai.
''')

st.header('Preprocessing')
st.write('''
pertama akan dicek apakah terdapat data missing pada dataset. digunakan kode berikut  \n
         dat.isna().sum()

''')
sat=dat.isna().sum()
sat
st.write('''
jadi dapat dilihat bahwa tidak terdapat data missing  \n
''')


st.write('''
selanjutnya akan dilkakukan encoding, akan digunakan one hot encoding. berikut kode yang digunakan  \n
         attributes = ["brand", "price", "weight", "crust", "cheese", "size", "toppings", "spicy"]
    X = pd.get_dummies(dat[attributes], drop_first=True)
    X = X.apply(lambda col: col.astype(float) if col.dtype == bool else col)
    y = dat["ranking"]
    X.head()
''')
attributes = ["brand", "price", "weight", "crust", "cheese", "size", "toppings", "spicy"]

# One-hot encode ALL attributes, including price and weight (since they are discrete levels)
X = pd.get_dummies(dat[attributes], drop_first=True)

# Ensure all boolean columns are converted to numeric (0/1)
X = X.apply(lambda col: col.astype(float) if col.dtype == bool else col)

# Dependent variable (rank)
y = dat["ranking"]

X
st.write('''
selanjutnya akan ditambahkan variabel constant sebagai intercept untuk atribut.  \n
         X_const = sm.add_constant(X)
''')
X_const = sm.add_constant(X)
st.write('''
terus karena kemakin kecil rangking lebih disukai pembeli, maka nilai rangking akan di transform supaya semakin besar nilai lebih disukai pembeli.  \n
         y_transformed = max(y) + 1 - y
''')
y_transformed = max(y) + 1 - y

st.header('conjoint analysis')
st.write('''
pertama akan dibikin modelnya.   \n
         model = sm.OLS(y_transformed, X_const).fit()
         print(model.summary())
''')
model = sm.OLS(y_transformed, X_const).fit()
st.write(model.summary())
st.write('''
selanjutnya akan dilihat Part-Worth Utilities dari setiap level atribut.  \n
         part_worths = pd.DataFrame({
        "Attribute_Level": X_const.columns,
        "Utility": model.params.values
    }).sort_values(by="Utility", ascending=False)

    part_worths = part_worths[part_worths["Attribute_Level"] != "const"]
         
    plt.figure(figsize=(10,6))
    sns.barplot(data=part_worths, x='Utility', y='Attribute_Level', palette='viridis')
    plt.title('Part-Worth Utilities (Higher = More Preferred)')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('Utility')
    plt.ylabel('Attribute Level')
    plt.show()
''')

part_worths = pd.DataFrame({
    "Attribute_Level": X_const.columns,
    "Utility": model.params.values
}).sort_values(by="Utility", ascending=False)

# Remove intercept
part_worths = part_worths[part_worths["Attribute_Level"] != "const"]

plot1=plt.figure(figsize=(10,6))
sns.barplot(data=part_worths, x='Utility', y='Attribute_Level', palette='viridis')
plt.title('Part-Worth Utilities (Higher = More Preferred)')
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Utility')
plt.ylabel('Attribute Level')
st.pyplot(plot1)

st.write('''
berdasarkan hasil grafik hal yang dipalingsukai pembeli sebagai berikut:  
- untuk weight, pelanggan paling suka pizza berat 400g dan paling surang mengukai pizza berat 100g, jadi dapat dibilang bahwa pelangan lebih suka pizza yang lebih berat.  
- untuk crust, pembeli paling suka crust thin dan paling tidak suka crust thick, jadi pembeli lebih suka crust thin dibanding crust thick
- untuk topping, pembeli paling suka topping panner dan paling tidak suka topping mushroom, jadi pembeli lebih suka topping panner dibanding topping mushroom
- untuk price, pembeli paling suka price $4 dan paling tidak suka price $1, jadi pembeli lebih suka pizza dengan harga yang lebih mahal.
- untuk brand, pembeli paling suka brand oven story dan paling tidak suka brand pizzahut, jadi berikut brand yang disukai pembeli ovenstory > onesta== domino > pizzahut.
- untuk cheese, pembeli paling suka cheese chedar dan plaing tidak suka cheese mozzarella, jadi pembeli lebih suka pizza yang menggunakan keju chedar dibanding keju mozzarella.
- untuk size, pembeli paling suka size large dan paling tidak suka size medium, jadi para pembeli leih suka pizza yang lebih besar.
- unuk spicy, pembeli paling suka spicy level normal dan paling tidak suka spicy level extra, jadi pembeli lebih suka pizza yang kurang pedas.
''')

st.write('''
selanjutnya akan dilihat Relative Importance setiap atribut.   \n
         utility_dict = dict(zip(part_worths['Attribute_Level'], part_worths['Utility']))


    base_levels = {
        'brand_Dominos': 0, 'price_1.0': 0, 'weight_100': 0,
        'crust_thick': 0, 'cheese_Cheddar': 0, 'size_large': 0,
        'toppings_mushroom': 0, 'spicy_extra': 0
    }

    for lvl, util in base_levels.items():
        if lvl not in part_worths['Attribute_Level'].values:
            part_worths = pd.concat([part_worths, pd.DataFrame([{
                'Attribute_Level': lvl, 'Utility': util
         }])], ignore_index=True)
        
    def get_attribute(col):
        return col.rsplit('_', 1)[0] if '_' in col else col

    part_worths['Attribute'] = part_worths['Attribute_Level'].apply(get_attribute)

    importance = (
        part_worths.groupby('Attribute')['Utility']
       .agg(lambda x: x.max() - x.min())
       .reset_index()
       .rename(columns={'Utility': 'Range'})
    )

    total = importance['Range'].sum()
    importance['Importance_%'] = (importance['Range'] / total * 100).round(1)

    importance2=importance.sort_values('Importance_%',ascending=False)
    y_pos = np.arange(len(importance2['Attribute']))
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(y_pos,importance2['Importance_%'])
    plt.xticks(y_pos,importance2['Attribute'])
    plt.tight_layout()
''')
utility_dict = dict(zip(part_worths['Attribute_Level'], part_worths['Utility']))

# add base levels = 0 manually
base_levels = {
    'brand_Dominos': 0, 'price_1.0': 0, 'weight_100': 0,
    'crust_thick': 0, 'cheese_Cheddar': 0, 'size_large': 0,
    'toppings_mushroom': 0, 'spicy_extra': 0
}


for lvl, util in base_levels.items():
    if lvl not in part_worths['Attribute_Level'].values:
        part_worths = pd.concat([part_worths, pd.DataFrame([{
            'Attribute_Level': lvl, 'Utility': util
        }])], ignore_index=True)
        
def get_attribute(col):
    return col.rsplit('_', 1)[0] if '_' in col else col

part_worths['Attribute'] = part_worths['Attribute_Level'].apply(get_attribute)

importance = (
    part_worths.groupby('Attribute')['Utility']
    .agg(lambda x: x.max() - x.min())
    .reset_index()
    .rename(columns={'Utility': 'Range'})
)

total = importance['Range'].sum()
importance['Importance_%'] = (importance['Range'] / total * 100).round(1)

importance2=importance.sort_values('Importance_%',ascending=False)

y_pos = np.arange(len(importance2['Attribute']))
plot2=plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(y_pos,importance2['Importance_%'])
plt.xticks(y_pos,importance2['Attribute'])
plt.tight_layout()
st.pyplot(plot2)

st.write('''
Jadi dari hasil feature importance, didapatkan bahwa feature weight merupakan feature yang paling penting bagi pembeli, selanjutnya jenis crust, terus jenis topping, terus price pizza dan tingkat kepedasan memiliki importance yang sama, dan terakhir 
brand, jenis keju dan size pizza juga memiliki importance yang sama.  \n
         ''')

st.write('''
jadi pizza optimal bedasarkan hasil conjoint analysis adalah sebagai berikut
- weight: 400g
- crust: thin
- topping: paneer
- spicy: normal
- price: $4
- brand: oven story
- cheese: chedar
- size: large
''')

st.write('''
Jadi berdasarkan hasil analisis, diberikan rekomendasi berikut:
- karena pembeli suka pizza yang lebih berat, mungkin bisa dicoba membuat pizza yang memiliki berat yang lebih tinggi
- karena weight memiliki feature importance yang sangat tinggi, mungkinbisa menambah kan opsi pada attribut lain seperti topping atau crust, supaya pembeli tidak hanya mementingkan berat pizza
- karena size pizza kurang penting pada pembeli, maka dapat dibuat supaya hanya menjual pizza pada satu size saja, supaya pengembangan dapat difokuskan pada bagian lain saja.
''')
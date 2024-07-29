import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from PIL import Image


icon = Image.open('icon.png')

st.set_page_config(layout = 'wide' ,
                   page_title='Sabahirr' ,
                   page_icon = icon)

# Title and description
st.title("Marketing Campaign Analysis")
st.text("This app analyzes marketing campaigns and visualizes the results.")

df = pd.read_csv('marketing.csv', parse_dates=['date_served', 'date_subscribed', 'date_canceled'])

df['converted'] = df['converted'].astype('bool')
df['is_retained'] = df['is_retained'].astype('bool')

# Mapping for channels
channel_dict = {'House Ads': 1, 'Instagram': 2, 'Facebook': 3, 'Email': 4, 'Push': 5}
df['channel_code'] = df['marketing_channel'].map(channel_dict)

# Add the new column is_correct_lang
df['is_correct_lang'] = np.where(df["language_displayed"] == df["language_preferred"], "Yes", "No")

# Add a DoW column
df['DoW'] = df["date_subscribed"].dt.dayofweek


def conversion_rate(dataframe, column_names):
    column_conv = dataframe[dataframe["converted"] == True].groupby(column_names)["user_id"].nunique()
    column_total = dataframe.groupby(column_names)["user_id"].nunique()
    conversion_rate = round((column_conv / column_total),2)
    conversion_rate = conversion_rate.fillna(0)
    return conversion_rate.reset_index()

def retention_rate(dataframe, column_names):
    column_ret = dataframe[dataframe["is_retained"] == True].groupby(column_names)["user_id"].nunique()
    column_total = dataframe.groupby(column_names)["user_id"].nunique()
    retention_rate = round((column_ret / column_total),2)
    retention_rate = retention_rate.fillna(0)
    return retention_rate.reset_index()

def lift(a, b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    lift = (a_mean - b_mean) / b_mean
    return lift

menu = st.sidebar.selectbox('', ['Conversion and Retention rates' , 'A/B testing', 'Results'])

Marketing_channel = df[['marketing_channel']]
Language_displayed = df['language_displayed'] 
Age_group = df['age_group']

#st.sidebar.button('Calculate'):
if menu =='Conversion and Retention rates':

    st.sidebar.header('Conversion and Retention rates')
    if st.sidebar.markdown("""
        <style>
        .button {
            background-color: #C0C0C0; 
            border: none; 
            color: white; 
            padding: 15px 32px; 
            text-align: center;
            text-decoration: none; 
            display: inline-block; 
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer; 
            border-radius: 12px; 
        }
        .button:hover {
            background-color: #4CAF50; #C0C0C0/* Fare üzerine geldiğinde arka plan rengi */
        }
        </style>
        <a href="#" class="button">Calculate </a>
        """, unsafe_allow_html=True):

        column_conv = df[df["converted"] == True]["user_id"].nunique()
        column_total = df["user_id"].nunique()
        convers_rate = round((column_conv / column_total)*100,2)

        column_ret = df[df["is_retained"] == True]["user_id"].nunique()
        column_total = df["user_id"].nunique()
        retent_rate = round((column_ret / column_total)*100, 2)
        
        c_model1, c_model2 = st.columns(2)
        c_model1.subheader('Convertion Rate (%)')
        c_model1.write(convers_rate)
        c_model2.subheader('Retention Rate (%)')
        c_model2.write(retent_rate)


    st.sidebar.header('Conversion and retention rates for segments')

    num_category = st.sidebar.radio('Select the category quantity', ['One', 'Two'])

    if num_category == 'One':

        column = st.sidebar.selectbox('Select category', df.columns.tolist(), index = 2 )

        if column:
            column_conv = df[df["converted"] == True].groupby(column)["user_id"].nunique()
            column_total = df.groupby(column)["user_id"].nunique()
            conversion_rate = round((column_conv / column_total),2)
            conversion_rate = conversion_rate.fillna(0)
            
            conv_rate = pd.DataFrame(conversion_rate).reset_index()
            conv_rate['Conversion Rate']=conv_rate['user_id']
            conv_rate.drop(['user_id'], axis = 1, inplace= True)
            
            column_ret = df[df["is_retained"] == True].groupby(column)["user_id"].nunique()
            column_total = df.groupby(column)["user_id"].nunique()
            retention_rate = round((column_ret / column_total),2)
            retention_rate = retention_rate.fillna(0)
            ret_rate = pd.DataFrame(retention_rate).reset_index()
            ret_rate['Retention Rate']=ret_rate['user_id']
            ret_rate.drop(['user_id'], axis = 1, inplace= True)
            
            c_model1, c_model2 = st.columns(2)

            c_model1.subheader(f'Convertion Rate for {column}')
            c_model1.dataframe(conv_rate)
            c_model2.subheader(f'Retention Rate for {column}')
            c_model2.dataframe(ret_rate)


            c_model1, c_model2 = st.columns(2)
            
            if not conv_rate.empty:
                c_model1.subheader(f'Conversion rate for {column}(%)')
                plt.figure(figsize=(10, 6))
                sns.barplot(x=conv_rate[column], y=conv_rate['Conversion Rate'])
                plt.title('')
                plt.ylabel('Conversion Rate')
                plt.xlabel(column)
                for p in plt.gca().patches:
                    plt.gca().annotate(f'{p.get_height():.1%}', (p.get_x() * 1.005, p.get_height() * 1.005))
                c_model1.pyplot(plt)
            
            if not ret_rate.empty:
                c_model2.subheader(f'Retention rate for {column} (%)')
                plt.figure(figsize=(10, 6))
                sns.barplot(x=ret_rate[column], y=ret_rate['Retention Rate'])
                plt.title('')
                plt.ylabel('Retention Rate')
                plt.xlabel(column)
                for p in plt.gca().patches:
                    plt.gca().annotate(f'{p.get_height():.1%}', (p.get_x() * 1.005, p.get_height() * 1.005))
                c_model2.pyplot(plt)

    else:
        columns = st.sidebar.multiselect('Select two categories', df.columns.tolist(), max_selections=2)

        if len(columns) ==1:
            st.warning("Please enter two categories!")

        else:
            if columns:
                conv_rate = conversion_rate(df, columns).reset_index()
                conv_rate = pd.DataFrame(conv_rate)
                conv_rate['Conversion Rate']=conv_rate['user_id']
                conv_rate.drop(['user_id'], axis = 1, inplace= True)
                
                ret_rate = retention_rate(df, columns).reset_index()
                ret_rate = pd.DataFrame(ret_rate)
                ret_rate['Retention Rate']=ret_rate['user_id']
                ret_rate.drop(['user_id'], axis = 1, inplace= True)
                
                c_model1, c_model2 = st.columns(2)

                c_model1.subheader('Convertion Rate')
                c_model1.dataframe(conv_rate)
                c_model2.subheader('Retention Rate')
                c_model2.dataframe(ret_rate)
                
                if not conv_rate.empty:
                    st.subheader('Conversion rate (%)')
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=conv_rate, x=columns[0], y='Conversion Rate', hue=columns[1])
                    plt.title('')
                    plt.ylabel('Conversion Rate')
                    plt.xlabel('')
                    for p in plt.gca().patches:
                        plt.gca().annotate(f'{p.get_height():.1%}', (p.get_x() * 1.005, p.get_height() * 1.005))
                    st.pyplot(plt)
                
                if not ret_rate.empty:
                    st.subheader('Retention rate (%)')
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=ret_rate, x=columns[0], y='Retention Rate', hue=columns[1])
                    plt.title('')
                    plt.ylabel('Retention Rate')
                    plt.xlabel('')
                    for p in plt.gca().patches:
                        plt.gca().annotate(f'{p.get_height():.1%}', (p.get_x() * 1.005, p.get_height() * 1.005))
                    st.pyplot(plt)

elif menu == 'A/B testing':

    st.sidebar.header('A/B testing')
    st.subheader('A/B testing')
    
    if st.sidebar.markdown("""
        <style>
        .button {
            background-color: #C0C0C0; /* Yeşil arka plan */
            border: none; /* Kenarlık yok */
            color: white; /* Beyaz metin */
            padding: 15px 32px; /* Dolgu */
            text-align: center; /* Metin hizalama */
            text-decoration: none; /* Metin dekorasyonu yok */
            display: inline-block; /* Satır içi blok */
            font-size: 16px; /* Yazı tipi boyutu */
            margin: 4px 2px; /* Kenar boşlukları */
            cursor: pointer; /* İmleç tipi */
            border-radius: 12px; /* Kenar yuvarlama */
        }
        .button:hover {
            background-color: #4CAF50; #C0C0C0/* Fare üzerine geldiğinde arka plan rengi */
        }
        </style>
        <a href="#" class="button">Calculate A/B Test</a>
        """, unsafe_allow_html=True):
        
        subscribers = df.groupby(['user_id', 'variant'])["converted"].max().unstack(level=1)
        control = subscribers['control'].dropna().astype('int')
        personalization = subscribers['personalization'].dropna().astype('int')

        st.subheader('Lift')
        lift_value = lift(personalization, control)
        st.write(f'Lift: {lift_value:.3%}')

        t_stat, p_val = stats.ttest_ind(control, personalization)
        st.write(f'T-statistic: {t_stat:.3f}')
        st.write(f'P-value: {p_val:.3f}')

        if p_val < 0.05:
            st.write("There is a statistically significant difference between the two options.")
        else:
            st.write("There is no statistically significant difference between the two options.")
        
        

        c_model1, c_model2 = st.columns(2)
        
        c_model1.subheader('Conversion rate')
        conversion_rates = [control.mean(), personalization.mean()]
        labels = ['Control', 'Personalization']

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(conversion_rates, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('A/B Test Conversion Rate')
        
        c_model1.pyplot(fig)

        
        c_model2.subheader('Retention rate')
        retention_data = df.groupby(['user_id', 'variant'])["is_retained"].max().unstack(level=1)
        control_retention = retention_data['control'].dropna().astype('int')
        personalization_retention = retention_data['personalization'].dropna().astype('int')

        retention_rates = [control_retention.mean(), personalization_retention.mean()]
        labels_retention = ['Control', 'Personalization']

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(retention_rates, labels=labels_retention, autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('A/B Test Retention Rate')
        
        c_model2.pyplot(fig)



    st.subheader('A/B test for segments')

    st.sidebar.header('A/B test for segments')

    num_ab_category = st.sidebar.radio('Select the category quantity', ['One', 'Two'])


    if num_ab_category == 'One':
        columna = st.sidebar.selectbox('Select category', df.columns.tolist(), index = 2)

        if columna:
            ab_data = df.groupby(['user_id', 'variant', columna])["converted"].max().unstack(level=1)
            control = ab_data['control'].dropna().astype('int')
            personalization = ab_data['personalization'].dropna().astype('int')

            st.subheader(f'A/B Test Results for {columna}')

            lift_value = lift(personalization, control)
            st.write(f'Lift: {lift_value:.3%}')

            t_stat, p_val = stats.ttest_ind(control, personalization)
            st.write(f'T-statistic: {t_stat:.3f}')
            st.write(f'P-value: {p_val:.3f}')

            if p_val < 0.05:
                st.write("There is a statistically significant difference between the two options.")
            else:
                st.write("There is no statistically significant difference between the two options.")

            conv_rate = conversion_rate(df, [columna, 'variant'])
            ret_rate = retention_rate(df, [columna, 'variant'])

            st.subheader('Conversion rate and Retention rate for segments')
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))

            sns.barplot(data=conv_rate, x=columna, y='user_id', hue='variant', ax=ax[0])
            ax[0].set_title('Conversion Rate ')
            ax[0].set_ylabel('Conversion Rate')
            ax[0].set_xlabel(columna)

            sns.barplot(data=ret_rate, x=columna, y='user_id', hue='variant', ax=ax[1])
            ax[1].set_title('Retention Rate ')
            ax[1].set_ylabel('Retention Rate')
            ax[1].set_xlabel(columna)

            st.pyplot(fig)


    else :
    
        segment_columns = st.sidebar.multiselect('Select two categories', df.columns.tolist(), default=['marketing_channel'])

        if len(segment_columns)==1:
            st.warning("Please enter two categories!")

        else:

            for col in segment_columns:
                df[col] = df[col].astype(str)
            
            def ab_segmentation(segments):
                for segment_value in df[segments[0]].unique():
                    segment_df = df[df[segments[0]] == segment_value]
                    subscribers_segment = segment_df.groupby(['user_id', 'variant'])['converted'].max().unstack(level=1)
                    if 'control' in subscribers_segment.columns and 'personalization' in subscribers_segment.columns:
                        control_segment = subscribers_segment['control'].dropna().astype('int')
                        personalization_segment = subscribers_segment['personalization'].dropna().astype('int')
                        
                        lift_value = lift(personalization_segment, control_segment)
                        t_stat, p_val = stats.ttest_ind(control_segment, personalization_segment)
                        
                        st.write(f'{segments[0]}: {segment_value}')
                        st.write(f'Lift: {lift_value:.3%}')
                        st.write(f'T-statistic: {t_stat:.3f}')
                        st.write(f'P-value: {p_val:.3f}')

                    
                        fig, ax = plt.subplots(figsize=(10, 6))
                        width = 0.35
                        unique_values = segment_df[segments[1]].unique()
                        index = np.arange(len(unique_values))
                            
                        means_control = []
                        means_personalization = []
                            
                        for value in unique_values:
                            subset = segment_df[segment_df[segments[1]] == value]
                            control_mean = subset[subset['variant'] == 'control']['converted'].mean()
                            personalization_mean = subset[subset['variant'] == 'personalization']['converted'].mean()
                            means_control.append(control_mean)
                            means_personalization.append(personalization_mean)
                            
                        bar1 = ax.bar(index, means_control, width, label='Control')
                        bar2 = ax.bar(index + width, means_personalization, width, label='Personalization')
                            
                        ax.set_xlabel(segments[1])
                        ax.set_ylabel('Conversion Rate')
                        ax.set_title(f' {segment_value}')
                        ax.set_xticks(index + width / 2)
                        ax.set_xticklabels(unique_values,rotation = 45)
                        ax.legend()
                            
                        for bar in bar1 + bar2:
                            height = bar.get_height()
                            ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
                            
                        st.pyplot(fig)
            
            if segment_columns:
                ab_segmentation(segment_columns)

else:
    pass




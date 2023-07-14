#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gradio as gr
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering
from transformers import pipeline


# In[ ]:


def get_answer(
    dropdown,
    question,
    view_as_table=False,
    model="google/tapas-finetuned-wtq",
    #progress=gr.Progress(),
):
    #progress(0, desc="Looking for answer in module guide...")
    df = pd.DataFrame()
    if dropdown == "Master Information Systems":
        df = pd.read_excel(
            r"03_extracted_final_modules/MS_IS_all_modules_orginal_15_rows_cleaned.xlsx"
        )
    elif dropdown == "Bachelor Information Systems":
        df = pd.read_excel(r"03_extracted_final_modules/BA_IS_all_modules_15.xlsx")
    elif dropdown == "Bachelor Management":
        df = pd.read_excel(r"03_extracted_final_modules\BA_MM_all_modules_15.xlsx")
    df = df.astype(str)
    print(question)
    question = str(question)
    print(df.shape)
    question = [question]

    if model == "google/tapas-finetuned-wtq":
        tqa = pipeline(
            task="table-question-answering", model="google/tapas-base-finetuned-wtq"
        )
    elif model == "google/tapas-large-finetuned-wtq":
        tqa = pipeline(
            task="table-question-answering", model="google/tapas-large-finetuned-wtq"
        )

    results = tqa(table=df, query=question)
    print(results)
    cells_input = results["cells"]
    cells_input = str(cells_input)
    cells_input = cells_input.replace("[", "")
    cells_input = cells_input.replace("]", "")
    cells_input = cells_input.replace("'", "")

    print(cells_input)
    print(results)
    html_string_short = f"<h1>Short Answer:</h1><p>{cells_input}</p>"
    row_numbers = [coord[0] for coord in results["coordinates"]]
    df_short = df.iloc[row_numbers]
    df_short = df_short.dropna(axis=1, how="all")
    df_short = df_short.loc[:, (df_short != "--").any(axis=0)]
    html_table = (
        f"<hr><h2>Complete Module(s):</h2><p>{df_short.to_html(index=False)}</p>"
    )

    # check if there are more than 1 rows in df_short
    html_string = ""
    if df_short.shape[0] > 1 or view_as_table == True:
        html_string = html_table
    elif df_short.shape[0] == 1:
        html_string = """<html>
        <head>
            <style>
                .module-info {
                    margin-bottom: 20px;
                }
                .module-info h3, .module-info p {
                    margin: 0;
                }
            </style>
        </head>
        <body>
            <hr>
            <h1>Detailed Module Information</h1>

            <div class="module-info">
                <h3>Module title:</h3>
                <p>Project Seminar</p>
            </div>
            <div class="module-info">
                <h3>Abbreviation:</h3>
                <p>12-PS-192-m01</p>
            </div>
            <div class="module-info">
                <h3>Module coordinator:</h3>
                <p>Faculty of Business Management and Economics</p>
            </div>
            <div class="module-info">
                <h3>Module offered by:</h3>
                <p>Holder of the Chair of Business Management and Business</p>
            </div>
            <div class="module-info">
                <h3>ETCS:</h3>
                <p>15</p>
            </div>
            <div class="module-info">
                <h3>Method of grading:</h3>
                <p>numerical grade</p>
            </div>
            <div class="module-info">
                <h3>Duration:</h3>
                <p>1 semester</p>
            </div>
            <div class="module-info">
                <h3>Module level:</h3>
                <p>graduate</p>
            </div>
            <div class="module-info">
                <h3>Contents:</h3>
                <p>In small project teams of 4 to 10 members, students will spend several months actively working on a specific and realistic problem with practical relevance. They will progress through several project stages including as-is analysis, to-be conception and implementation of an IS solution. The project teams will be required to work independently and will only receive advice and minor support from research assistants.</p>
            </div>
            <div class="module-info">
                <h3>Intended learning outcomes:</h3>
                <ul>
                    <li>Analyze business tasks and requirements and generate fitting IS solutions</li>
                    <li>Apply project management methods</li>
                    <li>Internalize stress, time and conflict management by means of practical teamwork</li>
                </ul>
            </div>
            <div class="module-info">
                <h3>Courses:</h3>
                <p>Project: preparing a conceptual design (approx. 150 hours), designing and implementing an approach to solution (approx. 300 hours) as well as presentation (approx. 20 minutes), weighted 1:2:1</p>
                <p>Language of assessment: German, English</p>
                <p>Creditable for bonus</p>
            </div>
            <div class="module-info">
                <h3>Workload:</h3>
                <p>450 hours</p>
            </div>
        </body>
        </html>"""
    else:
        html_string = ""

    return html_string_short, html_string


def change_html_link(dropdown_item):
    html_link = ""
    if dropdown_item == "Master Information Systems":
        html_link = f'<p>View complete pdf here:  <a href="https://www2.uni-wuerzburg.de/mhb/MHB1-en-88-i45-H-2018.pdf" target="_blank">{dropdown_item}</a></p> <p>Ask whatever you want to know about the module guide here. You can ask formality-based and content-based questions.</p>'
    elif dropdown_item == "Bachelor Information Systems":
        html_link = f'View complete pdf here:  <a href="https://www2.uni-wuerzburg.de/mhb/MHB1-en-82-277-H-2021.pdf" target="_blank">{dropdown_item}</a></p> <p>Ask whatever you want to know about the module guide here. You can ask formality-based and content-based questions.</p>'
    elif dropdown_item == "Bachelor Management":
        html_link = f'View complete pdf here: <a href="https://www2.uni-wuerzburg.de/mhb/MHB1-en-82-184-H-2008.pdf" target="_blank">{dropdown_item}</a></p> <p>Ask whatever you want to know about the module guide here. You can ask formality-based and content-based questions.</p>'

    return html_link


# In[20]:


with gr.Blocks() as demo:    
    gr.HTML(
    """
    <div style="text-align: center;">
        <img src="file/0915NC_Studienplaetze.jpg" alt="Module Guide Header Image" width="500">
    </div>
    """
    )

    gr.HTML(
        "<h1>Your Module Guide Assistant</h1>"
    )
    table = gr.Dropdown(
        [
            "Master Information Systems",
            "Bachelor Information Systems",
            "Bachelor Management",
        ],
        label="Select Module Guide",
        value="Master Information Systems",
    )
    html_link = gr.HTML(
        """
        <p>View complete PDF here: <a href="https://www2.uni-wuerzburg.de/mhb/MHB1-en-88-j10-H-2019.pdf" target="_blank">Master Information Systems</a></p>
        <p>Ask whatever you want to know about the module guide here. You can ask formality-based and content-based questions.</p>
    """
    )

    table.change(change_html_link, table, html_link)
    question = gr.Textbox(
        label="Question", value="How many ECTS credits does the project seminar have?"
    )
    with gr.Accordion("Advanced Options", open=False):
        with gr.Group():
            model_selction = gr.Dropdown(
                [
                    "google/tapas-finetuned-wtq",
                    "google/tapas-large-finetuned-wtq",
                ],
                label="Select Model",
                value="google/tapas-finetuned-wtq",
            )
            view_as_table_or_text = gr.Checkbox(
                label="View detailed information as table", value=False
            )

    ask_btn = gr.Button("Ask The Assistant")
    gr.HTML("<hr>")
    inputs = [table, question, view_as_table_or_text, model_selction]
    output_question = gr.HTML(label="Answer")
    outout_full_module = gr.HTML(label="Detailed Description")
    outputs = [output_question, outout_full_module]
    ask_btn.click(fn=get_answer, inputs=inputs, outputs=outputs, api_name="greet")

demo.launch(debug=True)


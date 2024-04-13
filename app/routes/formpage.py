from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

app = FastAPI()

@app.get("/")
async def root(request: Request):
    """
    请求root的时候，会向用户发送一个页面，页面上包含一个文本框，用户可以输入文本，点击提交按钮后，会将文本发送给后端，后端会返回一个相似度分数。
    """
    return templates.TemplateResponse("similarity-form.html", {"request": request})
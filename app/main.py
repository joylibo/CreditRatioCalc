from fastapi import FastAPI
from app.routes import formpage, mock, similarity, people

app = FastAPI()

# 注册路由
app.include_router(formpage.router)
app.include_router(mock.router)
app.include_router(similarity.router)
app.include_router(people.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




# @app.get("/get_score")
# def get_score():
#     """一个测试接口，用来测试数据库的连接
#     """
#     with Session(engine) as session:
#         # 查询前10行数据
#         query = select(ResidentCreditScore).limit(5)
#         results = session.exec(query)
#         return results.all()




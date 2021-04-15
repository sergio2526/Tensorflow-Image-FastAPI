import uvicorn
from fastapi import FastAPI, File, UploadFile
from prediction import predict, read_imagefile

app = FastAPI()


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Formatos de archivos permitidos jpg,jpeg,png"
    image = read_imagefile(await file.read())
    prediction = predict(image)

    return prediction



if __name__ == '__main__':
    uvicorn.run(app, port=8000 , host='0.0.0.0' )
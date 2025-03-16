from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceAdministrationClient, FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel
from collections import defaultdict

endpoint = "https://abijay.cognitiveservices.azure.com/"
key = "2qzVPrGbw0sEjDL48cdxQS0abZadNHkj3SasGZyoVVSGukmFdyU0JQQJ99BBACYeBjFXJ3w3AAAKACOGOJBM"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# database
groupDict = defaultdict(list)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print('Request for index page received')
    return templates.TemplateResponse('index.html', {"request": request})

@app.get('/favicon.ico')
async def favicon():
    file_name = 'favicon.ico'
    file_path = './static/' + file_name
    return FileResponse(path=file_path, headers={'mimetype': 'image/vnd.microsoft.icon'})

@app.post('/hello', response_class=HTMLResponse)
async def hello(request: Request, name: str = Form(...)):
    if name:
        print('Request for hello page received with name=%s' % name)
        return templates.TemplateResponse('hello.html', {"request": request, 'name':name})
    else:
        print('Request for hello page received with no name or blank name -- redirecting')
        return RedirectResponse(request.url_for("index"), status_code=status.HTTP_302_FOUND)
    
@app.post('/createGroup', response_class=HTMLResponse)
async def initialize(request: Request, groupId: str = Form(...), ):
    print('Request for initialize page received')
    with FaceAdministrationClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as face_admin_client:
        print(f"Create a large person group with id: {groupId}")
        face_admin_client.large_person_group.create(
            groupId, name="Class 1st", recognition_model=FaceRecognitionModel.RECOGNITION04
        )
    groupDict[groupId] = []
    return "Successfully created group with id: %s" % groupId

@app.post('/addPerson', response_class=HTMLResponse)
async def addPerson(request: Request, groupId: str = Form(...), personName: str = Form(...), imageFile: UploadFile = File(...)):
    print('Request for addPerson page received')
    with FaceAdministrationClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as face_admin_client:
        print(f"Create a student {personName} and add a face to him.")
        person_id = face_admin_client.large_person_group.create_person(
            groupId, name=personName, user_data="1"
        ).person_id
        face_admin_client.large_person_group.add_face(
            groupId,
            person_id,
            imageFile.file.read(),
            detection_model=FaceDetectionModel.DETECTION03,
            user_data="1",
        )
        groupDict[groupId].append([person_id, personName])
        print(f"Start to train the large person group: {groupId}.")
        poller = face_admin_client.large_person_group.begin_train(groupId)
        training_result = poller.result()
    return "Successfully added person %s to group with id: %s" % (personName, groupId)

@app.delete('/deleteGroup', response_class=HTMLResponse)
async def deleteGroup(request: Request, groupId: str = Form(...)):
    print('Request for deleteGroup page received')
    with FaceAdministrationClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as face_admin_client:
        print(f"Delete the large person group with id: {groupId}")
        face_admin_client.large_person_group.delete(groupId)
    del groupDict[groupId]    
    return "Successfully deleted group with id: %s" % groupId

@app.get('/getGroupInfo', response_class=HTMLResponse)
async def getGroupInfo(request: Request, groupId: str):
    print('Request for getGroupInfo page received')
    if groupId in groupDict:
        return "Successfully retrieved group with id: %s and members: %s" % (groupId, groupDict[groupId])
    else:
        return "No group found with id: %s" % groupId
    
@app.get('detectStudents', response_class=HTMLResponse)
async def detectStudents(request: Request, groupId: str = Form(...), imageFile: UploadFile = File(...)):
    with FaceClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as face_client:
        # Detect the face from the target image.
        detect_result = face_client.detect(
            imageFile.file.read(),
            detection_model=FaceDetectionModel.DETECTION03,
            recognition_model=FaceRecognitionModel.RECOGNITION04,
            return_face_id=True,
        )
        target_face_ids = list(f.face_id for f in detect_result)
        print(f"Detected {target_face_ids} face(s) in the target image.")

        result = face_client.identify_from_large_person_group(
            face_ids=target_face_ids, large_person_group_id=groupId
        )
        for idx, r in enumerate(result):
            print(f"----- Identification result: #{idx+1} -----")
            print(f"{r.as_dict()}")
        return "Successfully identified students in the image with names: %s" % (groupId, [groupDict[groupId][i.person_id][1] for i in result if i.candidates and i.candidates[0].confidence > 0.5])

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)


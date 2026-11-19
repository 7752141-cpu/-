import os, base64, json, re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI(title='ROODELL Color AI')
ROOT = Path(__file__).parent
class ImageRequest(BaseModel):
    image: str
    mime: str = 'image/jpeg'
    codes: list[str] = []

@app.get('/')
async def home():
    return FileResponse(ROOT / 'static' / 'index.html')

@app.get('/health')
async def health():
    return {'ok': True, 'vision_configured': bool(os.getenv('OPENAI_API_KEY'))}

@app.post('/api/vision/{kind}')
async def vision(kind: str, req: ImageRequest):
    if kind not in ('recipe','price'):
        raise HTTPException(404, 'Unknown mode')
    if not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(503, 'На сервере не настроен OPENAI_API_KEY')
    if req.mime not in ('image/jpeg','image/png','image/webp'):
        raise HTTPException(400, 'Поддерживаются JPEG, PNG, WEBP')
    if len(req.image) > 14_000_000:
        raise HTTPException(413, 'Слишком большой файл')
    try:
        raw = base64.b64decode(req.image, validate=True)
    except Exception:
        raise HTTPException(400, 'Некорректное изображение')
    if len(raw) > 10_000_000:
        raise HTTPException(413, 'Изображение более 10 МБ')
    if kind == 'recipe':
        instructions = '''Ты распознаёшь рукописные рецепты автомобильной краски. Верни ТОЛЬКО JSON объект: {"color_code":"", "color_name":"", "parts":[{"code":"118","grams":116.4}], "warnings":[]}. Прочитай КАЖДУЮ строку; коды пигментов обычно 01,100,108,118,124,148,146,310,390,340 и т.п. Префиксы BC/CL/CLT не включай в code, но различай их при чтении. Десятичные разделители точка и запятая. Никаких придуманных значений: сомнительные строки вынеси в warnings, не подставляй цифры. Не перепутай цифры 0, 6, 8, 9. Известные коды прайса: ''' + ', '.join(req.codes)
    else:
        instructions = '''Распознай прайс автомобильных пигментов с фотографии. Верни ТОЛЬКО JSON объект {"items":[{"code":"118","name":"BC Базовый белый Challenger BC 3,5","pack":3.5,"price":1377892,"density":1}],"warnings":[]}. Price — стоимость всей упаковки в сумах с НДС, pack — литры из конца наименования (например BC 3,5 = 3.5 литра, BC 1 = 1 литр). НЕ путай колонку Количество (обычно 1 шт) с объёмом упаковки. Сохраняй названия, коды и цены. Не угадывай нечитабельные строки: добавь предупреждение. density=1 условно.'''
    try:
        client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        result = await client.chat.completions.create(model=os.getenv('OPENAI_VISION_MODEL','gpt-4.1-mini'), temperature=0, response_format={'type':'json_object'}, messages=[{'role':'system','content':instructions},{'role':'user','content':[{'type':'text','text':'Распознай все видимые строки точно, без догадок.'},{'type':'image_url','image_url':{'url':f'data:{req.mime};base64,{req.image}','detail':'high'}}]}], timeout=75)
        data = json.loads(result.choices[0].message.content)
        if kind == 'recipe':
            parts = []
            for p in data.get('parts', []):
                code = re.sub(r'\D','',str(p.get('code','')))
                grams = float(str(p.get('grams',0)).replace(',','.'))
                if code and 0 < grams < 100000:
                    parts.append({'code':code,'grams':grams})
            return {'parts':parts,'color_code':str(data.get('color_code',''))[:80],'color_name':str(data.get('color_name',''))[:120],'warnings':data.get('warnings',[])}
        items=[]
        for p in data.get('items',[]):
            code=re.sub(r'\D','',str(p.get('code','')))
            pack=float(str(p.get('pack',0)).replace(',','.'))
            price=float(str(p.get('price',0)).replace(',','.'))
            if code and pack>0 and price>=0:
                items.append({'code':code,'name':str(p.get('name',''))[:250], 'pack':pack,'price':price,'density':1})
        return {'items':items,'warnings':data.get('warnings',[])}
    except Exception as exc:
        raise HTTPException(502, 'Ошибка ИИ-распознавания. Повторите попытку или проверьте настройки сервера.') from exc
from supabase import create_client

@app.get('/api/db-check')
def db_check():
    try:
        db = create_client(
            os.environ['SUPABASE_URL'],
            os.environ['SUPABASE_SECRET_KEY']
        )
        db.table('recipes').select('*').limit(1).execute()
        return {'ok': True, 'database': 'connected'}
    except Exception as exc:
        print("DB CHECK ERROR:", type(exc).__name__,
              getattr(exc, "code", ""), flush=True)
        raise HTTPException(
            503, 'Database connection failed'
        )


import os
import secrets
import base64
from fastapi import Request
from fastapi.responses import Response

@app.middleware("http")
async def require_login(request: Request, call_next):
    username = os.getenv("APP_USERNAME")
    password = os.getenv("APP_PASSWORD")

    if not username or not password:
        return Response(
            "Authorization not configured",
            status_code=503
        )

    expected = "Basic " + base64.b64encode(
        f"{username}:{password}".encode("utf-8")
    ).decode("ascii")

    actual = request.headers.get("Authorization", "")

    if not secrets.compare_digest(
        actual.encode("utf-8"),
        expected.encode("utf-8")
    ):
        return Response(
            "Authorization required",
            status_code=401,
            headers={
                "WWW-Authenticate": 'Basic realm="ROODELL SERVICE"'
            }
        )

    return await call_next(request)


def get_db():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SECRET_KEY"]
    )


@app.get("/api/data")
def load_data():
    try:
        db = get_db()

        recipes = db.table("recipes").select(
            "*"
        ).execute().data

        items = db.table("recipe_items").select(
            "*"
        ).execute().data

        pigments = db.table("pigments").select(
            "*"
        ).execute().data

        settings = db.table("app_settings").select(
            "*"
        ).eq("id", "main").execute().data

        return {
            "ok": True,
            "recipes": recipes,
            "recipe_items": items,
            "pigments": pigments,
            "app_settings": settings[0] if settings else {}
        }

    except Exception as exc:
        print("LOAD DATA ERROR:", type(exc).__name__, flush=True)
        raise HTTPException(
            500, "Ошибка загрузки данных"
        )



from uuid import UUID
from pydantic import BaseModel, Field


class RecipeItemInput(BaseModel):
    pigment_code: str = Field(min_length=1)
    grams: float = Field(gt=0, allow_inf_nan=False)


class RecipeInput(BaseModel):
    id: UUID | None = None
    color_code: str = Field(min_length=1)
    color_name: str = Field(min_length=1)
    items: list[RecipeItemInput] = Field(min_length=1)


@app.post("/api/recipes/save")
def save_recipe(data: RecipeInput):
    try:
        db = get_db()

        result = db.rpc(
            "save_recipe",
            {
                "p_id": str(data.id) if data.id else None,
                "p_color_code": data.color_code,
                "p_color_name": data.color_name,
                "p_items": [
                    {
                        "pigment_code": item.pigment_code,
                        "grams": item.grams
                    }
                    for item in data.items
                ]
            }
        ).execute()

        return {
            "ok": True,
            "id": result.data
        }

    except Exception as exc:
        print(
            "SAVE RECIPE ERROR:",
            type(exc).__name__,
            flush=True
        )
        raise HTTPException(
            500, "Ошибка сохранения рецепта"
        )

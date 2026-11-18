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

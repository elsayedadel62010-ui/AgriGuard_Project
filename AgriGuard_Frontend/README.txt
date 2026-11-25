AGRI GUARD FRONTEND

Files:
index.html
 dashboard.html
css/style.css
js/auth.js
js/api.js
js/charts.js
js/dashboard.js
mock/mock-data.js
images/logo.png

FIRST TEST:
Open with VS Code Live Server. Dashboard starts with MOCK DATA.

REAL BACKEND:
Open js/api.js and change:
USE_MOCK: true
into:
USE_MOCK: false
Then replace API_BASE_URL and endpoint paths with the final backend contract.

Frontend only: no database, AI model, ESP32 code, or separate backend is included.

# Đối Diện — V1 + V2 + V3 AI

## Có gì trong bản này
- V1: giao diện chatbot hiện đại, responsive, progress, card lựa chọn.
- V2: sau lời khuyên, chatbot chuyển sang "làm cùng bạn": nhập một nhiệm vụ cụ thể và chia thành các bước nhỏ.
- V3: backend Node/Express + Gemini API (Google Gen AI SDK). AI có thể phân tích câu nhập tự nhiên, tạo kế hoạch và chọn câu chuyện từ STORY BANK.

## Chạy bản offline
Nếu chỉ muốn xem giao diện:
1. Mở `public/index.html` bằng trình duyệt.
2. Flow V1/V2 vẫn hoạt động; phần AI sẽ tự fallback.

## Chạy bản có AI
Cần Node.js 18+.
1. Đổi `.env.example` thành `.env`.
2. Điền `GEMINI_API_KEY`.
3. Có thể giữ `GEMINI_MODEL=gpt-5.6-luna`.
4. Chạy:
   `npm install`
   `npm start`
5. Mở `http://localhost:3000`

Không đặt API key trong frontend. API key phải ở server.

## Deploy để có link share
Deploy cả project Node này lên một hosting hỗ trợ Node.js. Sau khi deploy, người dùng chỉ cần mở URL của app. Biến môi trường `GEMINI_API_KEY` phải được cấu hình trên server.

## Lưu ý
STORY BANK đang là dữ liệu mẫu được viết ngắn gọn để tránh AI tự bịa câu chuyện. Khi làm bản chính thức, nên thay bằng các câu chuyện đã được kiểm chứng và thêm nguồn cho từng câu chuyện.


## Gemini
Project này dùng `@google/genai` ở backend, không đưa API key vào frontend. Google khuyến nghị dùng biến môi trường `GEMINI_API_KEY` cho API key và dùng SDK `@google/genai` cho JavaScript/Node.js. Model mặc định: `gemini-3.7-flash`. 

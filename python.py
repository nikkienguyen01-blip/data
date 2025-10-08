import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
from typing import List, Dict, Any

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

# --- Khởi tạo State cho Chat (Mới) ---
if "chat_messages" not in st.session_state:
    # Định dạng tin nhắn: [{"role": "user"|"assistant", "content": "..."}]
    st.session_state.chat_messages = []
if "chat_context" not in st.session_state:
    # Lưu trữ dữ liệu phân tích quan trọng làm ngữ cảnh cho AI
    st.session_state.chat_context = ""

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho NHẬN XÉT một lần ---
def get_ai_analysis(data_for_ai: str, api_key: str) -> str:
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét tổng quan."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm gọi API Gemini cho CHAT tương tác (MỚI) ---
def get_chat_response(messages: List[Dict[str, str]], context: str, api_key: str) -> str:
    """Gửi toàn bộ lịch sử chat và context đến Gemini API và nhận phản hồi."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # 1. Chuẩn bị System Instruction dựa trên Context
        system_instruction = f"""
        Bạn là một trợ lý phân tích tài chính chuyên nghiệp, thân thiện. 
        Hãy trả lời các câu hỏi của người dùng dựa trên dữ liệu Báo cáo Tài chính đã được cung cấp.
        Dữ liệu Báo cáo Tài chính cốt lõi (Tăng trưởng, Tỷ trọng, Chỉ số) là:
        {context}
        Hãy giữ các câu trả lời ngắn gọn, tập trung vào việc làm rõ các con số trong dữ liệu.
        """
        
        # 2. Chuyển đổi lịch sử chat Streamlit sang định dạng API
        api_messages = []
        for message in messages:
            api_messages.append({
                "role": "user" if message["role"] == "user" else "model",
                "parts": [{"text": message["content"]}]
            })
            
        # 3. Gọi API
        response = client.models.generate_content(
            model=model_name,
            contents=api_messages, # Gửi toàn bộ lịch sử để duy trì ngữ cảnh
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text

    except APIError as e:
        return f"Lỗi Chat API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong Chat: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")

            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (Dữ liệu này cũng được dùng làm Context cho Chat)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", 
                    f"{thanh_toan_hien_hanh_N}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            # Cập nhật context cho Chat sau khi phân tích thành công (Quan trọng cho Chức năng 6)
            st.session_state.chat_context = data_for_ai
            st.session_state.chat_messages = [] # Xóa lịch sử chat khi tải file mới

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                     st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

            # --- Chức năng 6: Khung Chat Tương tác với Gemini (MỚI) ---
            st.markdown("---")
            st.subheader("6. Khung Chat Tương tác (Hỏi & Đáp về Dữ liệu đã tải)")
            st.caption("Gemini sẽ trả lời các câu hỏi dựa trên dữ liệu Báo cáo Tài chính bạn vừa tải lên.")
            
            # Hiển thị lịch sử chat
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Xử lý input từ người dùng
            if prompt := st.chat_input("Hỏi Gemini về Báo cáo Tài chính đã tải..."):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if not api_key:
                    st.error("Lỗi Chat: Không tìm thấy Khóa API 'GEMINI_API_KEY'.")
                elif not st.session_state.chat_context:
                    st.error("Lỗi Chat: Dữ liệu phân tích chưa được tạo. Vui lòng tải file và chạy Phân tích trước.")
                else:
                    # 1. Thêm tin nhắn người dùng vào lịch sử và hiển thị
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # 2. Lấy phản hồi từ Gemini
                    with st.chat_message("assistant"):
                        with st.spinner("Gemini đang trả lời..."):
                            full_response = get_chat_response(
                                st.session_state.chat_messages, 
                                st.session_state.chat_context, 
                                api_key
                            )
                            st.markdown(full_response)
                    
                    # 3. Thêm tin nhắn của assistant vào lịch sử
                    st.session_state.chat_messages.append({"role": "assistant", "content": full_response})


    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
        print(f"Lỗi chi tiết: {e}") # Debugging
        st.session_state.chat_context = ""

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

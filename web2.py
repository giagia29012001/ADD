import os
import torch
import cv2
import json
import shutil
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
import pytesseract
from datetime import datetime, time,timedelta
import smtplib
from email.mime.text import MIMEText
from PIL import Image, ImageOps

@st.cache()
def load_model(path: str = 'weights/best.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = path)
    return model

@st.cache()
def load_model_swap(path: str='weights/best_swap.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = path)
    return model

@st.cache()
def load_model_inf(path: str='weights/best_inf.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = path)
    return model

@st.cache()
def find_point(xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    data = [{'x': x_center, 'y': y_center}]
    return data

@st.cache()
def load_file_structure(path: str = 'src/all_img.json') -> dict:
    with open(path, 'r') as f:
        return json.load(f)

@st.cache()
def load_list_of_images(
        all_images: dict,
        image_files_dtype: str,
        diseases_species: str
        ) -> list:
    species_dict = all_images.get(image_files_dtype)
    list_of_files = species_dict.get(diseases_species)
    return list_of_files

@st.cache(allow_output_mutation = True)
def get_prediction(img_bytes, model):
    results = model(img_bytes)  
    return results

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def send_email(subject, body, recipients):
    sender = "gia.nguyen2001tp@hcmut.edu.vn"
    password = "xmdcubnbbippoxod"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(sender, password)
        smtp_server.sendmail(sender, recipients, msg.as_string())

def main():
    st.set_page_config(
        page_title = "Automatic Detection of Dermatological diseases",
        page_icon = "🔎",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

    model = load_model()
    model_swap = load_model_swap()
    model_inf = load_model_inf()
    all_images = load_file_structure()
    types_of_diseases = sorted(list(all_images['train'].keys()))
   
    dtype_file_structure_mapping = {
    'Images Used To Train The Model': 'train',
    'Images Used To Tune The Model': 'valid',
    'Images The Model Has Never Seen': 'test'
}

    data_split_names = list(dtype_file_structure_mapping.keys())


    with st.sidebar:
        st.image(Image.open('src/LOGOMINI.jpg'), width = 100)

        select_page = st.radio("CHỌN", ["TRANG CHỦ","CHUẨN ĐOÁN DA LIỄU", "HỦY LỊCH ĐÃ ĐẶT","VỀ ADD", "LIÊN HỆ"])
        st.markdown("<br /><br /><br /><br /><br /><br />", unsafe_allow_html = True)
        st.markdown("<hr />", unsafe_allow_html = True)

    if select_page == "CHUẨN ĐOÁN DA LIỄU":
        col1, col2 = st.columns([8.1, 4])
        file_img, file_vid, key_path, web_cam = '', '', '',''

        with col2:
            logo = Image.open('src/LOGOBIG.jpg')
            st.image(logo, use_column_width = True)

            with st.expander("Cách sử dụng ADD?", expanded = True):
                 with open('src/title/STORY.md', 'r', encoding='utf-8') as file:
                     markdown_text = file.read()

                  # Hiển thị nội dung markdown
            st.markdown(markdown_text, unsafe_allow_html=True)


        with col1:
            with open('src/title/INFO.md', 'r', encoding='utf-8') as file:
                 markdown_text = file.read()

                  # Hiển thị nội dung markdown
            st.markdown(markdown_text, unsafe_allow_html=True)

            choice_way = st.radio("Chọn một", ["Tải ảnh lên","Tải video lên","Sử dụng webcam","Chọn từ ảnh có sẵn"])

            if choice_way == "Tải ảnh lên":
                file_img = st.file_uploader('Tải một hình ảnh về bệnh da liễu')

                if file_img:
                    img = Image.open(file_img)

            elif choice_way == "Tải video lên":
                file_vid = st.file_uploader('Tải một video về bệnh da liễu')
                if file_vid:
                    frame_skip = 20 # display every 300 frames
                    st.video(file_vid)
                    vid = file_vid.name
                    with open(vid, mode='wb') as f:
                        f.write(file_vid.read()) # save video to disk
     

            elif choice_way == "Sử dụng webcam":
                
                start_button = st.button("Bắt đầu mở webcam")
                frame_skip = 100
                stframe=st.empty()
                if start_button:
                    web_cam = cv2.VideoCapture(0)
                    vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
                    output = cv2.VideoWriter("cam_video.mp4", vid_cod, 20.0, (640, 480))
                    stop_button = st.button("Dừng webcam")

                    
                    while True:
                        ret, frame = web_cam.read()
                        if not ret:
                            st.warning("Failed to read frame from webcam.")
                            break
                        
                        stframe.image(frame, channels="BGR")
                        output.write(frame)
                        
                        if stop_button:
                            break
                    
                    web_cam.release()
                    output.release()
                    cv2.destroyAllWindows()

                vid='cam_video.mp4'

            else:

                dataset_type = st.selectbox("Loại dữ liệu", data_split_names)
                data_folder = dtype_file_structure_mapping[dataset_type]

                selected_species = st.selectbox("Loại bệnh da liễu",types_of_diseases)
                available_images = load_list_of_images(all_images, data_folder, selected_species)
                image_name = st.selectbox("Tên hình ảnh", available_images)

                key_path = os.path.join('dermatological_diseases_dataset', data_folder, image_name)
                img = cv2.imread(key_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    

            recipe_button = st.button('Lấy kết quả!')

        st.markdown("<hr />", unsafe_allow_html = True)

        if recipe_button:
            with st.spinner("Chờ trong giây lát..."):
                if file_img or key_path:
                    col3, col4 = st.columns([5, 4])
                    with col3: 
                        if os.path.isdir('./runs'):
                            shutil.rmtree('./runs')

                        results = get_prediction(img, model)
                        results.save()

                        img_res = cv2.imread('./runs/detect/exp/image0.jpg')
                        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                        st.header("Đây là kết quả phát hiện!")
                        st.image(img_res, use_column_width=True)

                        df = results.pandas().xyxy[0]
                        del df['class']
                        st.write(df)
                    with col4:
                        st.header("Mô tả")

                        des = set()
                        for name_type in df['name']:
                            if name_type not in des:
                                # Xử lý hiển thị mô tả cho từng loại bệnh
                                if name_type == 'muncoc':
                                       with st.expander("MỤN CÓC - U MỀM"):

                                            with open('src/title/MUNCOC.md', 'r', encoding='utf-8') as file:
                                                markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                            st.markdown(markdown_text, unsafe_allow_html=True)

                                
                                    
                                elif name_type == 'vaynen':
                                    with st.expander("VẨY NẾN- Á SỪNG"):

                                        with open('src/title/VAYNEN.md', 'r', encoding='utf-8') as file:
                                            markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                        st.markdown(markdown_text, unsafe_allow_html=True)
                                            

                                elif name_type == 'trungcado':
                                    with st.expander("TRỨNG CÁ ĐỎ"):

                                        with open('src/title/TRUNGCADO.md', 'r', encoding='utf-8') as file:
                                            markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                        st.markdown(markdown_text, unsafe_allow_html=True)

                                            

                                elif name_type == 'hacto':
                                    with st.expander("UNG THƯ HẮC TỐ"):

                                        with open('src/title/HACTO.md', 'r', encoding='utf-8') as file:
                                            markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                        st.markdown(markdown_text, unsafe_allow_html=True)

                                            
                                elif name_type == 'bachbien':
                                    with st.expander("BẠCH BIẾN"):

                                        with open('src/title/BACHBIEN.md', 'r', encoding='utf-8') as file:
                                            markdown_text = file.read()

                                            # Hiển thị nội dung  markdown
                                        st.markdown(markdown_text, unsafe_allow_html=True)

                                des.add(name_type)

                        if not des:
                            st.info("Không có dữ liệu để mô tả!")

                elif vid:
                    vid_cap = cv2.VideoCapture(vid)
                    cur_frame = 0
                    success = True

                    while success:
                        ret, frame = vid_cap.read()  # get next frame from video
                        if not ret:
                            print("Failed to read frame from video.")
                            break
                        
                        if cur_frame % frame_skip == 0:  # only analyze every n=300 frames
                            print('frame: {}'.format(cur_frame))
                            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(img)
                            
                            if os.path.exists('./runs'):
                                shutil.rmtree('./runs')
                            
                            results = get_prediction(pil_img, model)
                            results.save()

                            st.header("Đây là kết quả phát hiện!")

                            img_res = cv2.imread('./runs/detect/exp/image0.jpg')
                            if img_res is not None:
                                img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                                
                                col5, col6 = st.columns([5, 4])
                                with col5:
                                    st.image(img_res, use_column_width=True)

                                    df = results.pandas().xyxy[0]
                                    del df['class']
                                    st.write(df)
                                with col6:
                                    st.header("Mô tả")

                                    des = set()
                                    for name_type in df['name']:
                                        if name_type not in des:
                                            # Xử lý hiển thị mô tả cho từng loại bệnh
                                            if name_type == 'muncoc':
                                                with st.expander("MỤN CÓC - U MỀM"):

                                                        with open('src/title/MUNCOC.md', 'r', encoding='utf-8') as file:
                                                            markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                        st.markdown(markdown_text, unsafe_allow_html=True)

                                            
                                                
                                            elif name_type == 'vaynen':
                                                with st.expander("VẨY NẾN- Á SỪNG"):

                                                    with open('src/title/VAYNEN.md', 'r', encoding='utf-8') as file:
                                                        markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                    st.markdown(markdown_text, unsafe_allow_html=True)
                                                        

                                            elif name_type == 'trungcado':
                                                with st.expander("TRỨNG CÁ ĐỎ"):

                                                    with open('src/title/TRUNGCADO.md', 'r', encoding='utf-8') as file:
                                                        markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                    st.markdown(markdown_text, unsafe_allow_html=True)

                                                        

                                            elif name_type == 'hacto':
                                                with st.expander("UNG THƯ HẮC TỐ"):

                                                    with open('src/title/HACTO.md', 'r', encoding='utf-8') as file:
                                                        markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                    st.markdown(markdown_text, unsafe_allow_html=True)

                                                        
                                            elif name_type == 'bachbien':
                                                with st.expander("BẠCH BIẾN"):

                                                    with open('src/title/BACHBIEN.md', 'r', encoding='utf-8') as file:
                                                        markdown_text = file.read()

                                                        # Hiển thị nội dung  markdown
                                                    st.markdown(markdown_text, unsafe_allow_html=True)

                                            des.add(name_type)

                                    if not des:
                                        st.info("Không có dữ liệu để mô tả!")
                        cur_frame += 1
                            
                else:
                    st.error('Không có dữ liệu. Vui lòng chọn một hình ảnh hoặc video về bệnh da liễu!')

    elif select_page == "TRANG CHỦ":
            col1, col2 = st.columns([8.1, 4])
            file_img, file_img_bhyt ='',''
            type = ""

            with col2:
                logo = Image.open('src/LOGOBIG.jpg')
                st.image(logo, use_column_width = True)

                with st.expander("Cách sử dụng ADD?", expanded = True):
                    with open('src/title/STORY1.md', 'r', encoding='utf-8') as file:
                        markdown_text = file.read()

                    # Hiển thị nội dung markdown
                st.markdown(markdown_text, unsafe_allow_html=True)

            with col1:
                with open('src/title/INFO2.md', 'r', encoding='utf-8') as file:
                    markdown_text = file.read()

                    # Hiển thị nội dung markdown
                st.markdown(markdown_text, unsafe_allow_html=True)

                mail = st.text_input('Để tiện cho việc liên hệ khi có bất kì sự thay đổi nào. Vui lòng nhập mail')

                st.title("Đặt lịch")

                current_date = datetime.now()

                # Tính toán ngày giới hạn là 7 ngày kế tiếp
                max_date = current_date + timedelta(days=7)

                # Hiển thị trình điều khiển ngày với giới hạn
                selected_date = st.date_input("Chọn ngày:", current_date, max_value=max_date)

                choice_way = st.radio("Chọn lịch khám bệnh", ["Buổi sáng","Buổi chiều"])
                if choice_way == "Buổi sáng":
                # Trình điều khiển thời gian
                    start_time = time(8, 0)
                    end_time = time(11, 0)
                else: 

                    # Giới hạn thời gian buổi chiều từ 12 giờ trưa đến 17 giờ trưa
                    start_time = time(13, 0)
                    end_time = time(17, 0)

                # Tạo danh sách thời gian 
                time_range = [start_time.replace(hour=h, minute=0) for h in range(start_time.hour, end_time.hour + 1)]

                # Trình điều khiển thời gian dưới dạng dropdown cho buổi sáng
                selected_time = st.selectbox("Chọn giờ :", time_range)

                date_str = selected_date.strftime("%Y-%m-%d")
                hour_str = selected_time.strftime('%H')

                # Tạo đường dẫn tập tin
                directory = f'host/booking/{date_str}/{hour_str}'
                file_path = f'{directory}/booking.txt'

            # Tạo thư mục nếu chưa tồn tại
                if os.path.exists(directory):
                    st.warning("Tệp tin đã tồn tại. Hãy chọn một thời gian khác.")
                
                file_img = st.file_uploader('Để hỗ trợ quá trình thăm khám bệnh. Vui lòng tải một hình ảnh hiện trạng da của bạn để chúng tôi đưa ra những chuẩn đoán ban đầu')
                if file_img:
                    img = Image.open(file_img) 
                file_img_bhyt = st.file_uploader('Vui lòng cung cấp thông tin bệnh nhân bằng cách tải một hình ảnh bảo hiểm y tế của bệnh nhân trên ứng dụng VISSID')    
                if file_img_bhyt:
                    img_bhyt = Image.open(file_img_bhyt)
    
                recipe_button = st.button('Gửi thông tin.')
                if recipe_button:
                    st.markdown("<hr />", unsafe_allow_html = True)
                    with st.spinner("Chờ trong giây lát..."):
                        if mail and file_img and file_img_bhyt and selected_date and selected_time:
                            if file_img:
                                if os.path.isdir('./runs'):
                                    shutil.rmtree('./runs')

                                    results = get_prediction(img, model)
                                    results.save()

                                    img_res = cv2.imread('./runs/detect/exp/image0.jpg')

                                    df = results.pandas().xyxy[0]
                                    name_type=df['name'].item()
                                    des = set()
                                    for name_type in df['name']:
                                        if name_type not in des:
                                                # Xử lý hiển thị mô tả cho từng loại bệnh
                                            if name_type == 'muncoc':
                                                type='Kết quả chuẩn đoán sơ bộ: Mụn cóc'
                    
                                            elif name_type == 'vaynen':
                                                type='Kết quả chuẩn đoán sơ bộ: Vẩy nến'

                                            elif name_type == 'trungcado':
                                                type='Kết quả chuẩn đoán sơ bộ: Trứng cá đỏ' 

                                            elif name_type == 'hacto':
                                                type='Kết quả chuẩn đoán sơ bộ: Hắc tố'
                                                            
                                            elif name_type == 'bachbien':
                                                type='Kết quả chuẩn đoán sơ bộ: Bạch biến'

                                            des.add(name_type)

                                        if not des:
                                            type='Không có dữ liệu để mô tả!'
                                        st.write(type)

                                else:
                                    st.error('Không có dữ liệu. Vui lòng chọn một hình ảnh hiện trạng trên da')

                                if file_img_bhyt:
                                    if os.path.isdir('./runs'):
                                        shutil.rmtree('./runs')
                                    results = get_prediction(img_bhyt, model_swap)
                                    results.save()
                                    img_res = cv2.imread('./runs/detect/exp/image0.jpg')
                                    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                                    df = results.pandas().xyxy[0]
                                    del df['class']
                                    des = set()
                                    for name_type in df['name']:
                                        if name_type not in des:
                                            if name_type == 'qr':
                                                id_rows = df[df['name'] == 'qr']
                                                    # Lặp qua từng hàng trong DataFrame với 'name_type' là 'qr'
                                                for index, row in id_rows.iterrows():
                                                    x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                                                    data_qr=find_point(x_min, y_min, x_max, y_max)
                                                    x_qr, y_qr = data_qr[0]['x'], data_qr[0]['y']

                                            if name_type == 'bhyt':
                                                id_rows = df[df['name'] == 'bhyt']
                                                    # Lặp qua từng hàng trong DataFrame với 'name_type' là 'bhyt'
                                                for index, row in id_rows.iterrows():
                                                    x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                                                    data_bhyt=find_point(x_min, y_min, x_max, y_max)
                                                    x_bhyt, y_bhyt = data_bhyt[0]['x'], data_bhyt[0]['y']                                
                                                    # Cắt ảnh
                                                    cropped_img = img_bhyt.crop((x_min, y_min, x_max, y_max))
                                                    # Hiển thị ảnh gốc và ảnh đã cắt
                                                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                                                    st.image(cropped_img, use_column_width=True)
                                    #so sánh tâm để xoay ảnh
                                    if x_qr > x_bhyt and y_qr < y_bhyt:
                                        rotated_image = cropped_img.rotate(180, expand=True)

                                    elif x_qr < x_bhyt and y_qr > y_bhyt:
                                        rotated_image = cropped_img

                                    elif x_qr > x_bhyt and y_qr > y_bhyt:
                                        rotated_image = cropped_img.rotate(-90, expand=True)

                                    else:
                                        rotated_image = cropped_img.rotate(90, expand=True)
                                    #ảnh sau khi xoay
                                    st.image( rotated_image, use_column_width=True)

                                    if os.path.exists('./runs'):
                                                shutil.rmtree('./runs')
                                            
                                    results = get_prediction(rotated_image, model_inf)
                                    results.save()

                                    st.header("Đây là kết quả phát hiện!")

                                    img_res = cv2.imread('./runs/detect/exp/image0.jpg')
                                    if img_res is not None:

                                        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                                        st.image(img_res, use_column_width=True)

                                        df = results.pandas().xyxy[0]
                                        del df['class']
                                        st.write(df)
                                        st.header("Thông tin trích xuất từ ảnh BHYT")

                                        des = set()
                                        for name_type in df['name']:
                                            if name_type not in des:
                                                            # Xử lý hiển thị mô tả cho từng loại bệnh
                                                if name_type == 'id':
                                                    id_rows = df[df['name'] == 'id']

                                                    # Lặp qua từng hàng trong DataFrame với 'name_type' là 'id'
                                                    for index, row in id_rows.iterrows():
                                                        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                                                            # Cắt ảnh
                                                        cropped_img = rotated_image.crop((x_min, y_min, x_max, y_max))
                                                        border_size = 10
                                                        # Thêm viền đen cho hình ảnh
                                                        cropped_img = ImageOps.expand(cropped_img, border=border_size, fill='white')
                                                        image=cropped_img

                                                        # Trích xuất văn bản sử dụng pytesseract
                                                        text = pytesseract.image_to_string(image)
                                                        text_id="MÃ BHYT: "+text
                                                        # Hiển thị kết quả
                                                        st.write(text_id)
                                                if name_type == 'name':
                                                    id_rows = df[df['name'] == 'name']

                                                    # Lặp qua từng hàng trong DataFrame với 'name_type' là 'id'
                                                    for index, row in id_rows.iterrows():
                                                        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                                                            # Cắt ảnh
                                                        cropped_img = rotated_image.crop((x_min, y_min, x_max, y_max))
                                                        border_size = 10
                                                        # Thêm viền đen cho hình ảnh
                                                        cropped_img = ImageOps.expand(cropped_img, border=border_size, fill='white')
                                                        image=cropped_img
                                                        # Trích xuất văn bản sử dụng pytesseract
                                                        text = pytesseract.image_to_string(image)
                                                        text_name= "HỌ VÀ TÊN: "+text

                                                        # Hiển thị kết quả
                                                        st.write(text_name)
                                                if name_type == 'birth':
                                                    id_rows = df[df['name'] == 'birth']

                                                    # Lặp qua từng hàng trong DataFrame với 'name_type' là 'id'
                                                    for index, row in id_rows.iterrows():
                                                        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                                                            # Cắt ảnh
                                                        cropped_img = rotated_image.crop((x_min, y_min, x_max, y_max))
                                                        border_size = 10
                                                        # Thêm viền đen cho hình ảnh
                                                        cropped_img = ImageOps.expand(cropped_img, border=border_size, fill='white')
                                                        image=cropped_img

                                                        # Trích xuất văn bản sử dụng pytesseract
                                                        text = pytesseract.image_to_string(image)
                                                        text_birth= "NGÀY THÁNG NĂM SINH: "+text
                                                        # Hiển thị kết quả
                                                        st.write(text_birth)
                                                if name_type == 'sex':
                                                    id_rows = df[df['name'] == 'sex']

                                                    # Lặp qua từng hàng trong DataFrame với 'name_type' là 'id'
                                                    for index, row in id_rows.iterrows():
                                                        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                                                            # Cắt ảnh
                                                        cropped_img = rotated_image.crop((x_min, y_min, x_max, y_max))
                                                        border_size = 30
                                                        # Thêm viền đen cho hình ảnh
                                                        cropped_img = ImageOps.expand(cropped_img, border=border_size, fill='white')
                                                        image=cropped_img
                                                        # Trích xuất văn bản sử dụng pytesseract
                                                        text = pytesseract.image_to_string(image)
                                                        if text: 
                                                        # Hiển thị kết quả
                                                            text_sex= "GIỚI TÍNH: "+text
                                                        else: 
                                                            text_sex="GIỚI TÍNH : Nữ"
                                                        st.write(text_sex)

                                                        # Hiển thị kết quả
                                                if name_type == 'place':
                                                    id_rows = df[df['name'] == 'place']

                                                    # Lặp qua từng hàng trong DataFrame với 'name_type' là 'id'
                                                    for index, row in id_rows.iterrows():
                                                        x_min, y_min, x_max, y_max = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                                                            # Cắt ảnh
                                                        cropped_img = rotated_image.crop((x_min, y_min, x_max, y_max))
                                                        border_size = 10
                                                        # Thêm viền đen cho hình ảnh
                                                        cropped_img = ImageOps.expand(cropped_img, border=border_size, fill='white')
                                                        image=cropped_img

                                                        # Trích xuất văn bản sử dụng pytesseract
                                                        text = pytesseract.image_to_string(image)
                                                        text_place= "NƠI KHÁM CHỮA BỆNH BAN ĐẦU: "+text
                                                        # Hiển thị kết quả
                                                        st.write(text_place)
                                else:
                                    st.error('Không có dữ liệu. Vui lòng chọn một hình ảnh bảo hiểm y tế trên ứng dụng VssID')

                                os.makedirs(directory)
                                file_booking = f"Ngày đặt lịch: {selected_date}\nGiờ đặt lịch: {selected_time.strftime('%H:%M')}\n{type}"
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.write(file_booking)
                                img.save(f'{directory}/diseases_image.jpg')
                                file_mail = f'{directory}/{mail}.txt'
                                with open( file_mail, "w", encoding="utf-8") as f:
                                    f.write(mail)
                                subject = "XÁC NHẬN ĐĂNG KÍ LỊCH KHÁM BỆNH"
                                body= "Bạn đã đăng kí lịch khám bệnh thành công trên website ADD.\n"+read_file(file_path)+"\nVui lòng kiểm tra thông tin bên trên và phản hồi về cho chúng tôi qua mục liên hệ khi có bất cứ nhầm lẫn nào"
                                recipients=mail
                                send_email(subject, body, recipients)
                                st.success("Chúng tôi đã gửi thông tin xác nhận đăng kí về mail. Vui lòng kiểm tra mail!")
                        else:
                            st.warning('Vui lòng cung cấp đầy đủ thông tin.')


    elif select_page == "HỦY LỊCH ĐÃ ĐẶT":
            col1, col2 = st.columns([8.1, 4])
            with col2:
                logo = Image.open('src/LOGOBIG.jpg')
                st.image(logo, use_column_width = True)

                with st.expander("Cách sử dụng ADD?", expanded = True):
                    with open('src/title/STORY2.md', 'r', encoding='utf-8') as file:
                        markdown_text = file.read()

                    # Hiển thị nội dung markdown
                st.markdown(markdown_text, unsafe_allow_html=True)

            with col1:
                with open('src/title/INFO3.md', 'r', encoding='utf-8') as file:
                    markdown_text = file.read()

                    # Hiển thị nội dung markdown
                st.markdown(markdown_text, unsafe_allow_html=True)

                mail = st.text_input('Vui lòng nhập mail bạn đã đăng kí với chúng tôi')
                current_date = datetime.now()

                # Tính toán ngày giới hạn là 7 ngày kế tiếp
                max_date = current_date + timedelta(7)

                # Hiển thị trình điều khiển ngày với giới hạn
                selected_date = st.date_input("Chọn ngày:", current_date, max_value=max_date)

                choice_way = st.radio("Chọn lịch khám bệnh", ["Buổi sáng","Buổi chiều"])
                if choice_way == "Buổi sáng":
                # Trình điều khiển thời gian
                    start_time = time(8, 0)
                    end_time = time(11, 0)
                else: 

                    # Giới hạn thời gian buổi chiều từ 12 giờ trưa đến 17 giờ trưa
                    start_time = time(13, 0)
                    end_time = time(17, 0)

                # Tạo danh sách thời gian 
                time_range = [start_time.replace(hour=h, minute=0) for h in range(start_time.hour, end_time.hour + 1)]

                # Trình điều khiển thời gian dưới dạng dropdown cho buổi sáng
                selected_time = st.selectbox("Chọn giờ :", time_range)

                date_str = selected_date.strftime("%Y-%m-%d")
                hour_str = selected_time.strftime('%H')

                # Tạo đường dẫn tập tin
                directory = f'host/booking/{date_str}/{hour_str}'
                file_path = f'{directory}/booking.txt'
                file_mail = f'{directory}/{mail}.txt'
                recipe_button = st.button('Gửi thông tin.')
                if recipe_button:
                    st.markdown("<hr />", unsafe_allow_html = True)
                    with st.spinner("Chờ trong giây lát..."):
                        if mail and selected_date and selected_time:
                            if os.path.exists(directory):
                                if os.path.exists(file_mail):
                                    subject = "XÁC NHẬN HỦY LỊCH KHÁM BỆNH"
                                    body= "Bạn hủy lịch khám bệnh thành công trên website ADD.\n"+read_file(file_path)+"\nVui lòng kiểm tra thông tin bên trên và phản hồi về cho chúng tôi qua mục liên hệ khi có bất cứ nhầm lẫn nào"
                                    recipients=mail
                                    send_email(subject, body, recipients)
                                    shutil.rmtree(directory)
                                    st.success('Chúng tôi đã gửi thông tin xác nhận hủy lịch về mail. Vui lòng kiểm tra mail!')

                                else: 
                                    st.warning('Vui lòng kiểm tra lại số điện thoại đã cung cấp. Chúng tôi chưa từng nhận lịch đăng kí cho số điện thoại này.')
                            else:
                                st.warning("Vui lòng kiểm tra lại thông tin đã cung cấp. Chúng tôi chưa từng nhận lịch đăng kí cho khoảng thời gian này.")
                        else:
                            st.warning('Vui lòng cung cấp đầy đủ thông tin.')
    elif select_page == "LIÊN HỆ":
        col1, col2 = st.columns([8.1, 4])
        file_img, file_vid, key_path,vip_cap = '', '', '',''

        with col2:
            st.image(Image.open('src/LOGOBIG.jpg'), width = 200)

            with st.expander("ĐỂ LIÊN HỆ VỚI ADD VUI LÒNG ĐIỀN FORM BÊN TRÁI", expanded = True):
                 with open('src/title/LIENHE.md', 'r', encoding='utf-8') as file:
                     markdown_text = file.read()

                  # Hiển thị nội dung markdown
            st.markdown(markdown_text, unsafe_allow_html=True)

        with col1:
            with open('src/title/INFO1.md', 'r', encoding='utf-8') as file:
                 markdown_text = file.read()

                  # Hiển thị nội dung markdown
            st.markdown(markdown_text, unsafe_allow_html=True)

            name = st.text_input('Họ và tên')
            mail = st.text_input('Email')
            tieude = st.text_input('Tiêu đề')
            noidung = st.text_input('Nội dung')

            recipe_button = st.button('Gửi')

            st.markdown("<hr />", unsafe_allow_html = True)

            if recipe_button:
                if name and mail and tieude and noidung:
                    email_content = f"Họ và tên: {name}\nEmail: {mail}\nTiêu đề: {tieude}\nNội dung: {noidung}"
        
                    # Lưu thông tin vào tệp tin trong thư mục "mess" với mã hóa utf-8
                    file_path = os.path.join("host/feedback", f"{name}_{mail}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(email_content)
                    st.success("Phản hồi của bạn đã được gửi thành công!")
                else:
                    st.error('Vui lòng điền đủ thông tin')

    else:
        logo = Image.open('src/GT.jpg')
        st.image(logo, use_column_width = True)
        with open('src/title/GT.md', 'r', encoding='utf-8') as file:
            markdown_text = file.read()

                  # Hiển thị nội dung markdown
        st.markdown(markdown_text, unsafe_allow_html=True)

if __name__ == '__main__':
    main()


import streamlit as st
import base64
import os
import mimetypes
import streamlit.components.v1 as components

def get_img_as_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"❌ Cannot find the image file: {image_path}")
        return None

def add_title_background(
    title_text,
    image_path="design/background.jpg",
    height=200,
    speed=30,
    opacity=0.6,
    grayscale=True
):
    # 确保 height 是数字
    if not isinstance(height, (int, float)):
        height = 200
    height = int(height)

    # 检查图片是否存在
    if not os.path.exists(image_path):
        st.error(f"❌ 图片文件不存在: {image_path}")
        st.title(title_text)
        return

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/jpeg'

    img_base64 = get_img_as_base64(image_path)
    if img_base64 is None:
        st.title(title_text)
        return

    filter_style = "filter: grayscale(100%) brightness(0.7);" if grayscale else ""

    html_code = f"""
    <div style="
        position: relative;
        height: {height}px;
        margin-bottom: 2rem;
        border-radius: 10px;
        overflow: hidden;
    ">
        <div id="bg-container" style="
            position: absolute;
            top: 0;
            left: 0;
            width: 200%;
            height: 100%;
            background-image: url('data:{mime_type};base64,{img_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: repeat-x;
            animation: scrollBg {float(speed):.1f}s linear infinite;
            opacity: {float(opacity):.2f};
            {filter_style}
        "></div>

        <div style="
            position: relative;
            z-index: 2;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            color: white;
            font-size: 2.5em;
            font-weight: bold;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
        ">{title_text}</div>

        <style>
        @keyframes scrollBg {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
        </style>
    </div>
    """

    components.html(html_code, height=height + 30, scrolling=False)

def add_custom_styles():
    """
    注入自定义 CSS 来优化滑块（Slider）的交互体验，使其更顺滑。
    """
    custom_css = """
    <style>
    /* 1. 让滑块的滑块（Thumb）移动时有过渡动画 */
    div.stSlider > div > div > div > div[role="slider"] {
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
        cursor: grab;
    }
    
    /* 2. 点击/拖动时的样式优化 */
    div.stSlider > div > div > div > div[role="slider"]:active {
        cursor: grabbing;
        transform: scale(1.1); /* 点击时稍微放大，增加反馈感 */
        box-shadow: 0px 0px 15px rgba(0, 153, 255, 0.6); /* 增加光晕 */
    }

    /* 3. 轨道（Track）的平滑过渡 */
    div.stSlider > div > div > div:first-child {
        transition: background-color 0.3s ease;
    }

    /* 4. 优化数值显示的字体，防止跳动 */
    div.stSlider p {
        font-variant-numeric: tabular-nums;
    }
    </style>
    """
    components.html(custom_css, height=0)
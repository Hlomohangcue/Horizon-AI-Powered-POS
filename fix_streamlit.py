import re

# Read the file
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace use_container_width=True with width='stretch'
content = re.sub(r'use_container_width=True', "width='stretch'", content)

# Write back to file
with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Fixed all use_container_width warnings')
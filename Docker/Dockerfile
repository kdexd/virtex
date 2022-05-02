# в репозитории написано что используется python 3.6
FROM python:3.6-stretch
MAINTAINER Kozlov <kkbimproject@gmail.com>

# устанавливаем параметры сборки
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# проверяем окружение python
RUN python3 --version
RUN pip3 --version

# задаем рабочую директорию для контейнера
WORKDIR  /usr/src/<virtex-d>

# устанавливаем зависимости python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# копируем все файлы из корня проекта в рабочую директорию
COPY src/ /src/
RUN ls -la /src/*

# запускаем приложение Python
CMD ["python3", "/src/setup.py"]
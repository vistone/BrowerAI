# 用于前端开发的Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app

# 复制package文件
COPY package*.json ./

# 安装依赖
RUN npm ci

# 复制源代码
COPY . .

# 构建应用
RUN npm run build

# 生产镜像
FROM node:20-alpine

WORKDIR /app

# 安装serve来提供静态文件
RUN npm install -g serve

# 从builder复制构建产物
COPY --from=builder /app/dist ./dist

# 暴露端口
EXPOSE 3000

# 启动应用
CMD ["serve", "-s", "dist", "-l", "3000"]

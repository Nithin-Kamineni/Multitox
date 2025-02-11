# Use an official Node.js image for the build and runtime
FROM node:18 as builder

WORKDIR /usr/src/app

# Copy package.json and package-lock.json
COPY package.json .
# Uncomment the next line if you have a package-lock.json
# COPY package-lock.json .

# Install dependencies
# RUN npm install --force

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PORT=3152

# Expose the port the app runs on
EXPOSE 3152

# Command to run the app
CMD ["npm", "start"]

cd blog_website
npm run build
if [ -d "../dist" ]; then
  rm -rf ../dist
  if [ $? -ne 0 ]; then
    echo "Failed to remove existing dist directory"
    exit 1
  fi
fi
mv dist ../dist

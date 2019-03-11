json.extract! image, :id, :filename, :created_at, :updated_at
json.url image_url(image, format: :json)

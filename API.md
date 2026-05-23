# API Documentation

Base URL: `https://your-domain.com/api` (or `http://localhost:8080/api` for local development)

All requests must include `Content-Type: application/json` header.

Authentication: Include `Authorization: Bearer {token}` header for protected endpoints.

## Response Format

All responses follow this format:

```json
{
  "code": 0,
  "message": "success",
  "data": {}
}
```

- `code`: 0 for success, non-zero for errors
- `message`: Human-readable message
- `data`: Response payload (null if error)

## Authentication Endpoints

### WeChat Login

```http
POST /auth/wx-login
Content-Type: application/json

{
  "code": "code_from_wx_login"
}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "userId": 123,
    "nickname": "User123",
    "avatarUrl": "https://...",
    "expiresIn": 604800
  }
}
```

### Get Current User

```http
GET /auth/me
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 123,
    "openid": "openid123",
    "nickname": "User123",
    "avatarUrl": "https://...",
    "status": 1,
    "createdAt": "2026-05-23T10:00:00",
    "updatedAt": "2026-05-23T10:00:00"
  }
}
```

## Music Endpoints

### List All Music (Paginated)

```http
GET /music/list?page=1&size=20
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Items per page (default: 20)

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "content": [
      {
        "id": 1,
        "title": "Song Title",
        "artist": "Artist Name",
        "album": "Album Name",
        "genre": "Pop",
        "durationSec": 180,
        "fileFormat": "mp3",
        "bitrate": 320,
        "fileSize": 7200000,
        "coverUrl": "https://...",
        "lyricUrl": "https://...",
        "createdAt": "2026-05-23T10:00:00",
        "updatedAt": "2026-05-23T10:00:00"
      }
    ],
    "totalElements": 100,
    "totalPages": 5,
    "currentPage": 1,
    "hasNext": true,
    "hasPrevious": false
  }
}
```

### Search Music

```http
GET /music/search?keyword=xxx&page=1&size=20
```

**Query Parameters:**
- `keyword`: Search term (title or artist name)
- `page`: Page number (default: 1)
- `size`: Items per page (default: 20)

**Response:** Same format as list endpoint

### Get Music Details

```http
GET /music/{musicId}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 1,
    "title": "Song Title",
    "artist": "Artist Name",
    "album": "Album Name",
    "genre": "Pop",
    "durationSec": 180,
    "fileFormat": "mp3",
    "bitrate": 320,
    "fileSize": 7200000,
    "coverUrl": "https://...",
    "lyricUrl": "https://...",
    "createdAt": "2026-05-23T10:00:00",
    "updatedAt": "2026-05-23T10:00:00"
  }
}
```

### Get Play URL

```http
GET /music/{musicId}/play-url
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "playUrl": "https://your-domain.com/music/2026/05/song.mp3",
    "expiresIn": 3600,
    "musicId": 1
  }
}
```

### Report Play Progress

```http
POST /music/{musicId}/report-progress?progressSec=120
Authorization: Bearer {token}
```

**Query Parameters:**
- `progressSec`: Current play position in seconds

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

## Favorites Endpoints

### Add to Favorites

```http
POST /favorites/{musicId}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

### Remove from Favorites

```http
DELETE /favorites/{musicId}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

### Get All Favorites

```http
GET /favorites
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": 1,
      "userId": 123,
      "musicId": 1,
      "createdAt": "2026-05-23T10:00:00"
    }
  ]
}
```

### Check if Music is Favorited

```http
GET /favorites/{musicId}/check
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": true
}
```

## Playlist Endpoints

### Create Playlist

```http
POST /playlists?name=My Playlist
Authorization: Bearer {token}
```

**Query Parameters:**
- `name`: Playlist name

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 1,
    "userId": 123,
    "name": "My Playlist",
    "isPublic": 1,
    "createdAt": "2026-05-23T10:00:00",
    "updatedAt": "2026-05-23T10:00:00"
  }
}
```

### Get User's Playlists

```http
GET /playlists
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": 1,
      "userId": 123,
      "name": "My Playlist",
      "isPublic": 1,
      "createdAt": "2026-05-23T10:00:00",
      "updatedAt": "2026-05-23T10:00:00"
    }
  ]
}
```

### Get Playlist Details

```http
GET /playlists/{playlistId}
Authorization: Bearer {token}
```

**Response:** Same as create playlist response

### Delete Playlist

```http
DELETE /playlists/{playlistId}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

### Add Music to Playlist

```http
POST /playlists/{playlistId}/music/{musicId}?sortOrder=1
Authorization: Bearer {token}
```

**Query Parameters:**
- `sortOrder`: Position in playlist (optional)

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

### Remove Music from Playlist

```http
DELETE /playlists/{playlistId}/music/{musicId}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": null
}
```

### Get Playlist Music

```http
GET /playlists/{playlistId}/music
Authorization: Bearer {token}
```

**Response:**
```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "id": 1,
      "playlistId": 1,
      "musicId": 1,
      "sortOrder": 0,
      "createdAt": "2026-05-23T10:00:00"
    }
  ]
}
```

## Error Responses

### Authentication Error (401)

```json
{
  "code": 401,
  "message": "Unauthorized",
  "data": null
}
```

### Not Found (404)

```json
{
  "code": 404,
  "message": "Resource not found",
  "data": null
}
```

### Server Error (500)

```json
{
  "code": 500,
  "message": "Internal server error",
  "data": null
}
```

## Rate Limiting

- **General endpoints**: 10 requests per second
- **Auth endpoints**: 5 requests per second
- **Burst**: Up to 20 requests allowed per second

Returns 429 (Too Many Requests) when exceeded.

## CORS Headers

All responses include:
```
Access-Control-Allow-Origin: * (or specific origin)
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: *
```

## Testing with cURL

```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8080/api/auth/wx-login \
  -H "Content-Type: application/json" \
  -d '{"code":"test"}' | jq -r '.data.token')

# Use token
curl -X GET http://localhost:8080/api/auth/me \
  -H "Authorization: Bearer $TOKEN"

# Search music
curl -X GET "http://localhost:8080/api/music/search?keyword=love&page=1&size=10"

# Add to favorites
curl -X POST http://localhost:8080/api/favorites/1 \
  -H "Authorization: Bearer $TOKEN"
```

## Testing with Postman

1. Create environment variable: `token`
2. In login endpoint "Tests" tab:
   ```javascript
   pm.environment.set("token", pm.response.json().data.token)
   ```
3. Use `{{token}}` in Authorization header for other requests

---

For more details, see DEVELOPMENT.md

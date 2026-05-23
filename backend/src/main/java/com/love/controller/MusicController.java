package com.love.controller;

import com.love.dto.ApiResponse;
import com.love.dto.MusicDTO;
import com.love.dto.PlayUrlResponse;
import com.love.service.MusicService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.Optional;

@Slf4j
@RestController
@RequestMapping("/music")
public class MusicController {
    private final MusicService musicService;

    public MusicController(MusicService musicService) {
        this.musicService = musicService;
    }

    @GetMapping("/list")
    public ResponseEntity<ApiResponse<Page<MusicDTO>>> listMusic(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size) {
        try {
            Page<MusicDTO> musics = musicService.listMusic(page, size);
            return ResponseEntity.ok(ApiResponse.success(musics));
        } catch (Exception e) {
            log.error("List music failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to list music"));
        }
    }

    @GetMapping("/search")
    public ResponseEntity<ApiResponse<Page<MusicDTO>>> searchMusic(
            @RequestParam String keyword,
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size) {
        try {
            Page<MusicDTO> musics = musicService.searchMusic(keyword, page, size);
            return ResponseEntity.ok(ApiResponse.success(musics));
        } catch (Exception e) {
            log.error("Search music failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to search music"));
        }
    }

    @GetMapping("/{id}")
    public ResponseEntity<ApiResponse<?>> getMusic(@PathVariable Long id) {
        try {
            Optional<MusicDTO> music = musicService.getMusic(id);
            if (music.isEmpty()) {
                return ResponseEntity.status(404)
                        .body(ApiResponse.error(404, "Music not found"));
            }
            return ResponseEntity.ok(ApiResponse.success(music.get()));
        } catch (Exception e) {
            log.error("Get music failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to get music"));
        }
    }

    @GetMapping("/{id}/play-url")
    public ResponseEntity<ApiResponse<?>> getPlayUrl(@PathVariable Long id) {
        try {
            Long userId = (Long) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
            PlayUrlResponse response = musicService.getPlayUrl(id, userId);
            return ResponseEntity.ok(ApiResponse.success(response));
        } catch (Exception e) {
            log.error("Get play URL failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, e.getMessage()));
        }
    }

    @PostMapping("/{id}/report-progress")
    public ResponseEntity<ApiResponse<?>> reportProgress(
            @PathVariable Long id,
            @RequestParam Integer progressSec) {
        try {
            Long userId = (Long) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
            musicService.reportPlayProgress(userId, id, progressSec);
            return ResponseEntity.ok(ApiResponse.success(null));
        } catch (Exception e) {
            log.error("Report progress failed", e);
            return ResponseEntity.status(500)
                    .body(ApiResponse.error(500, "Failed to report progress"));
        }
    }
}

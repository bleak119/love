package com.love.service;

import com.love.dto.MusicDTO;
import com.love.dto.PlayUrlResponse;
import com.love.entity.Music;
import com.love.entity.PlayHistory;
import com.love.repository.MusicRepository;
import com.love.repository.PlayHistoryRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Optional;

@Slf4j
@Service
public class MusicService {
    @Value("${upload.base-path}")
    private String uploadBasePath;

    private final MusicRepository musicRepository;
    private final PlayHistoryRepository playHistoryRepository;

    public MusicService(MusicRepository musicRepository, PlayHistoryRepository playHistoryRepository) {
        this.musicRepository = musicRepository;
        this.playHistoryRepository = playHistoryRepository;
    }

    public Page<MusicDTO> listMusic(int page, int size) {
        Pageable pageable = PageRequest.of(page - 1, size);
        return musicRepository.findByStatus((byte) 1, pageable).map(this::convertToDTO);
    }

    public Page<MusicDTO> searchMusic(String keyword, int page, int size) {
        Pageable pageable = PageRequest.of(page - 1, size);
        return musicRepository.searchByKeyword(keyword, pageable).map(this::convertToDTO);
    }

    public Optional<MusicDTO> getMusic(Long id) {
        return musicRepository.findById(id)
                .filter(m -> m.getStatus() == 1)
                .map(this::convertToDTO);
    }

    public PlayUrlResponse getPlayUrl(Long musicId, Long userId) {
        Optional<Music> music = musicRepository.findById(musicId);
        if (music.isEmpty() || music.get().getStatus() != 1) {
            throw new RuntimeException("Music not found");
        }

        // Record play history
        PlayHistory history = new PlayHistory();
        history.setUserId(userId);
        history.setMusicId(musicId);
        history.setPlayedAt(LocalDateTime.now());
        playHistoryRepository.save(history);

        // Return play URL (in production, this would generate a signed URL)
        String playUrl = music.get().getFilePath();
        if (music.get().getStorageType() == 1) {
            // OSS URL - add signed token
            playUrl = generateSignedUrl(playUrl);
        } else {
            // Local path - construct full URL
            playUrl = "http://your-domain/music/" + playUrl;
        }

        return new PlayUrlResponse(
                playUrl,
                3600L,  // 1 hour expiration
                musicId
        );
    }

    public void reportPlayProgress(Long userId, Long musicId, Integer progressSec) {
        // Update existing history or create new one
        PlayHistory history = new PlayHistory();
        history.setUserId(userId);
        history.setMusicId(musicId);
        history.setPlayedAt(LocalDateTime.now());
        history.setProgressSec(progressSec);
        playHistoryRepository.save(history);
    }

    private String generateSignedUrl(String filePath) {
        // TODO: Implement OSS signed URL generation
        return "https://your-oss.aliyuncs.com/" + filePath;
    }

    private MusicDTO convertToDTO(Music music) {
        return new MusicDTO(
                music.getId(),
                music.getTitle(),
                music.getArtist(),
                music.getAlbum(),
                music.getGenre(),
                music.getDurationSec(),
                music.getFileFormat(),
                music.getBitrate(),
                music.getFileSize(),
                music.getCoverUrl(),
                music.getLyricUrl(),
                music.getCreatedAt(),
                music.getUpdatedAt()
        );
    }
}

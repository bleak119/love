package com.love.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MusicDTO {
    private Long id;
    private String title;
    private String artist;
    private String album;
    private String genre;
    private Integer durationSec;
    private String fileFormat;
    private Integer bitrate;
    private Long fileSize;
    private String coverUrl;
    private String lyricUrl;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}

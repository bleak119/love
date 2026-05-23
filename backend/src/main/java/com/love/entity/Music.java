package com.love.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "music")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Music {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(length = 128)
    private String title;

    @Column(length = 128)
    private String artist;

    @Column(length = 128)
    private String album;

    @Column(length = 64)
    private String genre;

    @Column
    private Integer durationSec;

    @Column(length = 16)
    private String fileFormat;  // mp3, flac, wav

    @Column
    private Integer bitrate;

    @Column
    private Long fileSize;

    @Column
    private Byte storageType;  // 1: OSS/Object Storage, 2: Local

    @Column(length = 512)
    private String filePath;  // OSS key or local path

    @Column(length = 255)
    private String coverUrl;

    @Column(length = 255)
    private String lyricUrl;

    @Column(nullable = false)
    private Byte status = 1;  // 1: published, 0: unpublished

    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(nullable = false)
    private LocalDateTime updatedAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}

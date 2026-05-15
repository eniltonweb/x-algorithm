use crate::schema::{events::Event, tweet_events::TweetEvent};
use anyhow::{Context, Result};
use prost::Message;
use thrift::protocol::{TBinaryInputProtocol, TSerializable};
use xai_thunder_proto::InNetworkEvent;

const MAX_PAYLOAD_SIZE: usize = 1024 * 1024; // 1 MB

/// Deserialize a Thrift binary message into TweetEvent
pub fn deserialize_tweet_event(payload: &[u8]) -> Result<TweetEvent> {
    if payload.is_empty() {
        anyhow::bail!("Payload is empty");
    }
    if payload.len() > MAX_PAYLOAD_SIZE {
        anyhow::bail!("Payload size exceeds limit");
    }

    let mut cursor = std::io::Cursor::new(payload);
    let mut protocol = TBinaryInputProtocol::new(&mut cursor, true);

    TweetEvent::read_from_in_protocol(&mut protocol).context("Failed to deserialize TweetEvent")
}

/// Deserialize a Thrift binary message into Event
pub fn deserialize_event(payload: &[u8]) -> Result<Event> {
    if payload.is_empty() {
        anyhow::bail!("Payload is empty");
    }
    if payload.len() > MAX_PAYLOAD_SIZE {
        anyhow::bail!("Payload size exceeds limit");
    }

    let mut cursor = std::io::Cursor::new(payload);
    let mut protocol = TBinaryInputProtocol::new(&mut cursor, true);

    Event::read_from_in_protocol(&mut protocol).context("Failed to deserialize Event")
}

/// Deserialize a proto binary message into InNetworkEvent
pub fn deserialize_tweet_event_v2(payload: &[u8]) -> Result<InNetworkEvent> {
    if payload.is_empty() {
        anyhow::bail!("Payload is empty");
    }
    if payload.len() > MAX_PAYLOAD_SIZE {
        anyhow::bail!("Payload size exceeds limit");
    }

    InNetworkEvent::decode(payload).context("Failed to deserialize InNetworkEvent")
}
